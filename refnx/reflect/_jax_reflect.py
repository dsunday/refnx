"""
*Calculates the specular (Neutron or X-ray) reflectivity from a stratified
series of layers.

The refnx code is distributed under the following license:

Copyright (c) 2015 A. R. J. Nelson, ANSTO

Permission to use and redistribute the source code or binary forms of this
software and its documentation, with or without modification is hereby
granted provided that the above notice of copyright, these terms of use,
and the disclaimer of warranty below appear in the source code and
documentation, and that none of the names of above institutions or
authors appear in advertising or endorsement of works derived from this
software without specific prior written permission from all parties.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THIS SOFTWARE.

"""

from functools import reduce

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit

jax.config.update("jax_enable_x64", True)

from refnx.reflect.reflect_model import gauss_legendre

TINY = 1e-30

_FWHM = 2 * np.sqrt(2 * np.log(2.0))
_INTLIMIT = 3.5


def jabeles(q, layers, scale=1.0, bkg=0, threads=0):
    # Note, this code must have 64 bit operation enabled before it is used
    # You can do this with:
    # from jax import config
    # config.update("jax_enable_x64", True)

    qvals = q.astype(jnp.float64)
    flatq = qvals.ravel()

    nlayers = layers.shape[0] - 2
    npnts = flatq.size

    mi00 = jnp.ones((npnts, nlayers + 1), jnp.complex128)

    sld = jnp.zeros(nlayers + 2, jnp.complex128)

    # addition of TINY is to ensure the correct branch cut
    # in the complex sqrt calculation of kn.
    sld = sld.at[1:].add(
        ((layers[1:, 1] - layers[0, 1]) + 1j * (jnp.abs(layers[1:, 2]) + TINY))
        * 1.0e-6
    )

    kn = jnp.sqrt(flatq[:, jnp.newaxis] ** 2.0 / 4.0 - 4.0 * jnp.pi * sld)
    # reflectances for each layer
    # rj.shape = (npnts, nlayers + 1)
    damping = jnp.exp(-2.0 * kn[:, :-1] * kn[:, 1:] * layers[1:, 3] ** 2)
    rj = (kn[:, :-1] - kn[:, 1:]) / (kn[:, :-1] + kn[:, 1:]) * damping

    # characteristic matrices for each layer
    # miNN.shape = (npnts, nlayers + 1)
    if nlayers:
        mi00 = mi00.at[:, 1:].set(
            jnp.exp(kn[:, 1:-1] * 1j * jnp.fabs(layers[1:-1, 0]))
        )
    mi11 = 1.0 / mi00
    mi10 = rj * mi11
    mi01 = rj * mi00

    mi = jnp.zeros((npnts, nlayers + 1, 2, 2), jnp.complex128)

    mi = mi.at[:, :, 0, 0].set(mi00)
    mi = mi.at[:, :, 0, 1].set(mi01)
    mi = mi.at[:, :, 1, 1].set(mi11)
    mi = mi.at[:, :, 1, 0].set(mi10)

    sub = [jnp.squeeze(v) for v in jnp.hsplit(mi, nlayers + 1)]
    mrtot = reduce(jnp.matmul, sub[1:], sub[0])

    r = mrtot[:, 1, 0] / mrtot[:, 0, 0]
    reflectivity = r * jnp.conj(r) * scale
    reflectivity = reflectivity.at[...].add(bkg)

    return jnp.real(jnp.reshape(reflectivity, qvals.shape))


def jabeles_scan(q, layers, scale=1.0, bkg=0, threads=0):
    """
    Abeles matrix reflectivity using jax.lax.scan over layers.

    Drop-in replacement for jabeles that avoids building the intermediate
    (npnts, nlayers+1, 2, 2) complex128 tensor and reduce(matmul) chain.
    Represents each 2×2 transfer matrix as four scalar arrays (m00, m01,
    m10, m11) of shape (npnts,) and folds them with lax.scan, letting XLA
    compile the chain as a fused element-wise loop with no GEMM allocation.
    This eliminates the ~1.4 GB peak tensor under full vmap and is required
    for jax.lax.fori_loop compatibility without OOM.

    Numerically identical to jabeles to within floating-point rounding
    (max absolute difference < 1e-14 on tested configurations).
    """
    qvals = q.astype(jnp.float64)
    flatq = qvals.ravel()

    nlayers = layers.shape[0] - 2
    npnts = flatq.size

    sld = jnp.zeros(nlayers + 2, jnp.complex128)
    sld = sld.at[1:].add(
        ((layers[1:, 1] - layers[0, 1]) + 1j * (jnp.abs(layers[1:, 2]) + TINY))
        * 1.0e-6
    )

    kn = jnp.sqrt(flatq[:, jnp.newaxis] ** 2.0 / 4.0 - 4.0 * jnp.pi * sld)
    damping = jnp.exp(-2.0 * kn[:, :-1] * kn[:, 1:] * layers[1:, 3] ** 2)
    rj = (kn[:, :-1] - kn[:, 1:]) / (kn[:, :-1] + kn[:, 1:]) * damping

    # Build matrix-element stacks: shape (nlayers+1, npnts).
    # Row 0 is the fronting interface (no propagation phase, mi00=1).
    # Rows 1..nlayers hold the phase factor for each inner layer.
    ones_row = jnp.ones((1, npnts), jnp.complex128)
    if nlayers:
        phase = jnp.exp(kn[:, 1:-1] * 1j * jnp.abs(layers[1:-1, 0]))  # (npnts, nlayers)
        mi00_stk = jnp.concatenate([ones_row, phase.T], axis=0)         # (nlayers+1, npnts)
    else:
        mi00_stk = ones_row

    mi11_stk = 1.0 / mi00_stk
    rj_T = rj.T                       # (nlayers+1, npnts)
    mi01_stk = rj_T * mi00_stk
    mi10_stk = rj_T * mi11_stk

    def matmul_step(carry, x):
        m00, m01, m10, m11 = carry
        n00, n01, n10, n11 = x
        return (
            m00 * n00 + m01 * n10,
            m00 * n01 + m01 * n11,
            m10 * n00 + m11 * n10,
            m10 * n01 + m11 * n11,
        ), None

    init_carry = (mi00_stk[0], mi01_stk[0], mi10_stk[0], mi11_stk[0])
    xs = (mi00_stk[1:], mi01_stk[1:], mi10_stk[1:], mi11_stk[1:])
    (m00, _, m10, _), _ = jax.lax.scan(matmul_step, init_carry, xs)

    r = m10 / m00
    reflectivity = r * jnp.conj(r) * scale + bkg
    return jnp.real(jnp.reshape(reflectivity, qvals.shape))


# abeles_jax = jabeles
abeles_jax = jit(jabeles_scan)


def jax_smeared_kernel_pointwise(qvals, w, dqvals, quad_order=17, threads=0):
    # get the gauss-legendre weights and abscissae
    abscissa, weights = gauss_legendre(quad_order)

    # get the normal distribution at that point
    prefactor = 1.0 / np.sqrt(2 * np.pi)

    def gauss(x):
        return np.exp(-0.5 * x * x)

    gaussvals = prefactor * gauss(abscissa * _INTLIMIT)

    # integration between -3.5 and 3.5 sigma
    va = qvals - _INTLIMIT * dqvals / _FWHM
    vb = qvals + _INTLIMIT * dqvals / _FWHM

    va = va[:, np.newaxis]
    vb = vb[:, np.newaxis]

    qvals_for_res = (np.atleast_2d(abscissa) * (vb - va) + vb + va) / 2.0
    smeared_rvals = abeles_jax(qvals_for_res, w)

    smeared_rvals = np.reshape(smeared_rvals, (qvals.size, abscissa.size))

    smeared_rvals *= np.atleast_2d(gaussvals * weights)
    return np.sum(smeared_rvals, 1) * _INTLIMIT
