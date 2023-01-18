from tune.deconvolution import richardsonLucy
from tune.deconvolution.psf import PSF


def deconvolve(input_img, H, Ht, num_iters):
    input_shape = input_img.shape
    dec = richardsonLucy.dec_conv()
    dec.psf_calc(H, Ht, input_shape)
    return dec.deconv(input_img, lamb=1, num_iters=num_iters, weights=1).reshape(dec.shape)


def deconvolve_single(input_img, img_resolution, psf_invert, psf_path, psf_res, iterations):
    input_shape = input_img.shape

    psf = PSF(psf_path, psf_res)
    if psf_invert:
        psf.flip()
    psf.visualize("PSF extracted")

    psf.adjust_resolution(img_resolution)
    psf.resize(input_shape, [0, 0, 0])

    H, Ht = psf.calculate_otf()
    print("OTF calculated")

    dec = richardsonLucy.dec_conv()
    dec.psf_calc(H, Ht, input_shape)
    return dec.deconv(input_img, lamb=1, num_iters=iterations, weights=1).reshape(dec.shape)
