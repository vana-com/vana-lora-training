    # def canny(self, kwargs):
    #     # make a good canny edge image
    #     image = kwargs.get("image", None)
    #     image = np.array(image.convert("L"))  # make black and white np array
    #     height, width = kwargs.get("height"), kwargs.get("width")
    #     image = resize(image, (height, width))
    #     sigma = height // 175  # empirically
    #     sigma = 1  # empirically
    #     canny_im = feature.canny(
    #         image, sigma=sigma
    #     )  # , low_threshold=100, high_threshold=200)
    #     canny_im = np.stack([canny_im, canny_im, canny_im], axis=2)
    #     kwargs["image"] = Image.fromarray(canny_im.astype(np.uint8) * 255)