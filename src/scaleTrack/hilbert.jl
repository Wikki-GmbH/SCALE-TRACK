#=
    SPDX-License-Identifier: GPL-3.0-or-later

    This file is part of SCALE-TRACK, distributed under the GNU GPL v3 or
    later.  See scaleTrack.jl or <http://www.gnu.org/licenses/> for details.

    Copyright (C) 2026 Sergey Lesnik
    Copyright (C) 2026 Henrik Rusche
=#

#=
    The three-dimensional Hilbert curve.

    A curve of nBits resolution walks the (2^nBits)^3 points of a cube,
    visiting each exactly once, with consecutive points always neighbours.
    Those two properties are the whole of its usefulness: an interval of the
    curve stands for a compact region of the cube, so cutting the curve into
    equal intervals cuts the cube into equal and connected pieces.
=#

# One point of the curve, by Skilling's transform (AIP Conf. Proc. 707, 381):
# the curve index is spread over the three axes most significant bit first,
# Gray decoded, and the rotations and reflections of the recursive
# construction are undone bit plane by bit plane.
function hilbert_point(index::Unsigned, nBits::Integer)
    # Spread the index over the axes, three bits at a time
    x = y = z = zero(UInt64)
    for k in 0:(nBits - 1)
        triple = (index >> (3*(nBits - 1 - k))) & 0x7
        bit = nBits - 1 - k
        x |= ((triple >> 2) & 0x1) << bit
        y |= ((triple >> 1) & 0x1) << bit
        z |= (triple & 0x1) << bit
    end

    # Gray decode
    t = z >> 1
    z ⊻= y
    y ⊻= x
    x ⊻= t

    # Undo the rotations and reflections, coarsest bit plane first
    N = UInt64(2) << (nBits - 1)
    Q = UInt64(2)
    while Q != N
        P = Q - one(UInt64)
        if z & Q != 0
            x ⊻= P                              # invert
        else
            t = (x ⊻ z) & P                     # exchange
            x ⊻= t
            z ⊻= t
        end
        if y & Q != 0
            x ⊻= P
        else
            t = (x ⊻ y) & P
            x ⊻= t
            y ⊻= t
        end
        if x & Q != 0
            x ⊻= P
        end
        Q <<= 1
    end

    return (x, y, z)
end

# The number of points a curve of this resolution has
curve_length(nBits) = UInt64(1) << (3*nBits)
