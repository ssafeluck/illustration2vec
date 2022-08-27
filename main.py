import i2v
from PIL import Image


def main():
    illust2vec = i2v.make_i2v_with_chainer(
        "illust2vec_tag_ver200.caffemodel", "tag_list.json")

    # In the case of caffe, please use i2v.make_i2v_with_caffe instead:
    # illust2vec = i2v.make_i2v_with_caffe(
    #     "illust2vec_tag.prototxt", "illust2vec_tag_ver200.caffemodel",
    #     "tag_list.json")

    img = Image.open("images/miku.jpg")
    tags = illust2vec.estimate_plausible_tags([img], threshold=0.5)
    print(tags)


if __name__ == '__main__':
    main()
