def delete_white(file, file_out):

    from PIL import Image

    image = Image.open(file)

    width, height = image.size

    white_color = (255, 255, 255, 255)  # RGBA形式
    pixels = image.load()

    left = width
    right = 0
    top = height
    bottom = 0

    for x in range(width):
        for y in range(height):
            if pixels[x, y] != white_color:
                left = min(left, x)
                right = max(right, x)
                top = min(top, y)
                bottom = max(bottom, y)

    trimmed_image = image.crop((left, top, right + 1, bottom + 1))

    trimmed_image.save(file_out)
    