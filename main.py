#!/usr/bin/env python3

import argparse
import io
import json
import collections
import multiprocessing
import os
import glob
import traceback
import threading
import zipfile

import numpy

from PIL import Image, ImageChops, ImageMath, ImageFilter
from jxlpy import JXLImagePlugin
from tqdm import tqdm

Thumb = collections.namedtuple('Thumb', 'name key thumb tinygray')

def read_file(path, cache={}):
    if '\x00' in path:
        # zipfile
        zfname, path = path.split('\x00')
        key = (zfname, os.getpid())  # goofy multiprocessing hack
        if key not in cache:
            cache[key] = zipfile.ZipFile(zfname)
        return cache[key].read(path)
    return open(path, 'rb').read()


def load_image(path):
    im = Image.open(io.BytesIO(read_file(path)))
    if getattr(im, 'is_animated', None):
        raise ValueError("cannot handle animations for: " + path)
    if im.mode == '1':
        im = im.convert('RGB')
    elif im.mode == 'P':
        im = im.convert('RGBA')
    return im


def make_thumb(fname):
    im = load_image(fname)
    thumb = im.convert("RGB").resize((128, 128), resample=Image.LANCZOS)
    tinygray = thumb.resize((16, 16), resample=Image.LANCZOS).convert("L")
    return Thumb(fname, (im.mode, im.size), thumb, tinygray)


def barename(fname):
    return os.path.splitext(os.path.basename(fname.split('\x00')[-1]))[0]


BUCKETS = {}
NEAREST_COUNT = 3

def nearest_thumbs(thumb, n=None):
    n = n or NEAREST_COUNT
    others = [t for t in BUCKETS[thumb.key] if t.name != thumb.name]
    best = sorted(others, key=lambda t: diffcount(thumb.tinygray, t.tinygray))[:n * 3]
    return [thumb] + sorted(others, key=lambda t: diffcount(thumb.thumb, t.thumb))[:n]


def worker_init(buckets, nearest_count):
    # this is needed because the multiprocessing manager proxy object is too slow (locking?)
    global NEAREST_COUNT
    BUCKETS.update(buckets)
    NEAREST_COUNT = nearest_count


def diffcount(a, b):
    # return numpy.sum(numpy.square((numpy.asarray(a)-numpy.asarray(b)).astype(dtype=numpy.int8).astype(numpy.int32)))
    return numpy.sum((numpy.asarray(a)!=numpy.asarray(b)))


def compute_deltas(args):
    basename, targets, quality = args
    base = load_image(basename)
    bufs = [len(mkdelta(None, base, quality, 5))]
    for target in targets:
        try:
            bufs.append(len(mkdelta(target, base, quality, 4)))
        except Exception:
            print("error computing delta between", target, basename)
            traceback.print_exc()
            bufs.append(0)
    return bufs


def make_final(args):
    recipe, quality = args
    target, base = recipe.name, recipe.base
    return mkdelta(base, target, quality, 7)


def mkdelta(base, target, quality, effort):
    if isinstance(target, str):
        target = load_image(target)
    if base == None:
        img = target
    else:
        if isinstance(base, str):
            base = load_image(base)
        mask = ImageChops.difference(base, target).point(lambda p: p and 255).convert("L")
        if quality < 100:
            # dilate mask to avoid artifacts from lossy thin masks
            mask = mask.filter(ImageFilter.BoxBlur(4))
        else:
            # dilate slightly to avoid some patchier noise masks
            mask = mask.filter(ImageFilter.BoxBlur(2))
        mask = mask.point(lambda p: p and 1, "1")
        delta = Image.new("RGBA", base.size)
        delta = Image.composite(target, delta, mask)
        img = delta
    of = io.BytesIO()
    img.save(of, "jxl", quality=quality, effort=effort)
    return of.getvalue()


def quality_for(fname, quality):
    if quality > 0:
        return quality
    if os.path.splitext(fname)[1].lower() in ('.jpg', '.jpeg'):
        return 90
    return 100


def crunch(infiles, outdir, args):
    buckets = collections.defaultdict(list)
    thumbs = {}
    infiles = sorted(infiles)

    os.makedirs(os.path.dirname(outdir) or '.', exist_ok=True)

    bares = {}
    for fname in infiles:
        if '_ON_' in fname:
            raise ValueError('illegal filename (cannot contain "_ON_"): ' + fname)
        bare = barename(fname)
        if bare in bares:
            raise ValueError('bare filename collision between %r and %r' % (bares[bare], fname))
        bares[bare] = fname

    size_input = 0
    size_output = 0

    with multiprocessing.Pool(processes=args.threads) as pool:
        for thumb in tqdm(pool.imap_unordered(make_thumb, infiles), total=len(infiles), desc="generating thumbnails", unit='im'):
            buckets[thumb.key].append(thumb)
            thumbs[thumb.name] = thumb
            size_input += len(read_file(thumb.name))

    with multiprocessing.Pool(processes=args.threads, initializer=worker_init, initargs=(buckets, args.nearest)) as pool:
        nears = {}
        for thumbs in tqdm(pool.imap_unordered(nearest_thumbs, thumbs.values()),
                total=len(infiles), desc="grouping alike images", unit='im'):
            nears[thumbs[0].name] = [x.name for x in thumbs[1:]]

    with multiprocessing.Pool(processes=args.threads) as pool:
        # memory optimization: process connected components separately
        edges = collections.defaultdict(set)
        for a, bs in nears.items():
            edges[a].update(bs)
            for b in bs:
                edges[b].add(a)

        opts = collections.defaultdict(dict)

        with tqdm(total=sum(1+len(nears[x]) for x in infiles), desc="computing delta sizes", unit='im', smoothing=0) as pbar_delta:
            for fname, bufs in zip(infiles, pool.imap(compute_deltas, ((x, nears[x], quality_for(x, args.quality)) for x in infiles))):
                pbar_delta.update(len(bufs))
                pbar_delta.display()
                for near, bsize in zip([None] + nears[fname], bufs):
                    if bsize:
                        opts[fname][near] = bsize

        if args.verbose:
            print()
            for fname, bufs in sorted(opts.items()):
                print(barename(fname), ' '.join(f'{barename(n or "-")}:{len(b)}' for n,b in bufs.items()))
            print({k: {n: len(b) for n, b in v.items()} for k, v in opts.items()})

        if args.dump_dists:
            with open(args.dump_dists, 'w') as f:
                json.dump(opts, f)

        recipes = sorted(compute_best(opts, args.depth, verbose=args.verbose))

        zf = None
        if outdir.endswith('.zip'):
            zf = zipfile.ZipFile(outdir + '.tmp', 'w')
        else:
            os.makedirs(outdir, exist_ok=True)
        for recipe, buf in tqdm(
                zip(recipes, pool.imap(make_final, ((r, quality_for(r.name, args.quality)) for r in recipes))),
                total=len(infiles), desc="encoding final output", unit='im', smoothing=0):
            size_output += len(buf)
            if zf:
                zf.writestr(recipe.barename + '.jxl', buf, compresslevel=0)
            else:
                with open(os.path.join(outdir, recipe.barename + '.jxl'), 'wb') as f:
                    f.write(buf)
        if zf:
            zf.close()
            os.rename(outdir + '.tmp', outdir)

    print(f"input: {size_input/1024**2:,.2f} MiB, output: {size_output/1024**2:,.2f} MiB")


Recipe = collections.namedtuple('Recipe', 'name base barename size')


def compute_best(opts, max_depth, verbose=False):
    # Find the best delta patterns given the specified edges.
    # This is not optimal, but the heuristics are decent.

    class Node:
        def __init__(self, name, edges):
            self.name = name
            self.edges = edges
            self.best = edges[None]
            self.base = None

        def improvement(self, target):
            return max(0, self.best - self.edges[target.name])

        def update(self, target):
            if self.edges[target.name] < self.best:
                self.base = target
                self.best = self.edges[target.name]

        def chain(self):
            cur = self
            yield cur
            while cur.base != None:
                cur = cur.base
                yield cur

        def depth(self):
            return sum(1 for _ in self.chain())

        def __str__(self):
            return '_ON_'.join(barename(x.name) for x in self.chain())

    nodes = [Node(name, edges) for name, edges in opts.items()]
    todo = set(nodes)

    rev_nodes = collections.defaultdict(list)
    for node in todo:
        for edge in node.edges:
            rev_nodes[edge].append(node)

    def potential_improvement(node):
        return sum(other.improvement(node) for other in rev_nodes[node.name]) - node.best

    while todo:
        best = max(todo, key=potential_improvement)

        todo.remove(best)
        for edge in best.edges:
            rev_nodes[edge].remove(best)

        if verbose:
            print("picked", best, best.best, best.base, potential_improvement(best))
        if best.depth() < max_depth:
            for node in rev_nodes[best.name]:
                node.update(best)

    upper_bound = sum(max(n.edges.values()) for n in nodes)
    actual = sum(n.best for n in nodes)
    lower_bound = sum(min(n.edges.values()) for n in nodes)

    if verbose:
        print(f'{upper_bound=:,} {actual=:,} {lower_bound=:,}')

    for node in nodes:
        yield Recipe(node.name, node.base and node.base.name, str(node), node.edges[node.base and node.base.name])


def scan_zipfile(fname):
    with zipfile.ZipFile(fname) as zf:
        for name in zf.namelist():
            if name.endswith('/'):
                continue
            if name.endswith('.gif'):
                raise ValueError('cannot handle animations: ' + name)
            yield f'{fname}\x00{name}'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    parser.add_argument('output')
    parser.add_argument('-v', '--verbose', action='store_true', help="display more information")
    parser.add_argument('-n', '--limit', type=int, help="max number of images to transcode, useful for testing parameters")
    parser.add_argument('-k', '--nearest', type=int, help="number of similar images to test compression against", default=3)
    parser.add_argument('-q', '--quality', type=int, help="quality level, default auto (90 for jpg, 100 for png) (100=lossless)", default=-1)
    parser.add_argument('--depth', type=int, help="maximum number of images to decode to reconstruct (default 3)", default=3)
    parser.add_argument('--threads', type=int, help="number of parallel threads to use (default=number of CPU cores)", default=os.cpu_count())
    parser.add_argument('--dump-dists', help="output image distances to file for debugging")
    args = parser.parse_args()

    if os.path.splitext(args.input.lower())[1] in ('.zip', '.cbz'):
        pngs = sorted(scan_zipfile(args.input))
    else:
        pngs = sorted(glob.glob(args.input + '/*.*'))

    if args.limit:
        pngs = pngs[:args.limit]

    crunch(pngs, args.output, args)


if __name__ == '__main__':
    main()
