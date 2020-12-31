import ImageExplorer as imex
import JpgImageIO as imio

if __name__ == "__main__":
    image = imio.open_jpg_image("clean_eq_test/eq_1.jpg", bw=True)
    symbols, (width, height) = imex.explore_image(image)
    i = 0
    for symbol in symbols:
        symbol_img = imio.create_image_from_symbol_pixels_list(image, symbol[0])
        imio.save_jpg_image("symbol_extraction_test_folder/symbol_"+str(i)+".jpg", symbol_img)
        i += 1

