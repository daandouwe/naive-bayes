#!/usr/bin/env python
import argparse
import os


def main(args):
    indir = os.path.expanduser(args.indir)
    lines = []
    with open(indir) as f:
        for line in f.readlines():
            label, sentence = line.strip().split('|||')
            if int(label) == 2:
                continue
            elif int(label) < 2:
                lines.append(f'0 ||| {sentence.strip()}')
            else:
                lines.append(f'1 ||| {sentence.strip()}')
    outdir = os.path.expanduser(args.outdir)
    with open(outdir, 'w') as f:
        print('\n'.join(lines), file=f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('indir')
    parser.add_argument('outdir')
    args = parser.parse_args()

    main(args)
