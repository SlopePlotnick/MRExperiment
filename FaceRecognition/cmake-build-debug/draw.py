import matplotlib.pyplot as plt

# plt.figure()
#
# for i in range(10):
#     path = 'OUTPUT_EIGENFACES/eigenface' + '_' + str(i) + '.png'
#     image = plt.imread(path)
#     plt.subplot(2, 5, i + 1)
#     plt.imshow(image)
#     plt.axis('off')
#     plt.subplots_adjust(wspace = 0, hspace = 0)
# plt.show()

plt.figure()

l = []
cnt = 1
for i in range(10, 310, 15):
    path = 'OUTPUT_EIGENFACES/eigenface' + '_reconstruction_' + str(i) + '.png'
    image = plt.imread(path)
    plt.subplot(4, 5, cnt)
    plt.imshow(image, plt.cm.gray)
    plt.axis('off')
    # plt.subplots_adjust(wspace = 0, hspace = 0)
    plt.title(str(i))
    cnt = cnt + 1
plt.show()

# plt.figure()
#
# for i in range(16):
#     path = 'OUTPUT_FISHERFACES/fisherface' + '_' + str(i) + '.png'
#     image = plt.imread(path)
#     plt.subplot(4, 4, i + 1)
#     plt.imshow(image)
#     plt.axis('off')
#     # plt.subplots_adjust(wspace = 0, hspace = 0)
# plt.show()

# plt.figure()
#
# for i in range(16):
#     path = 'OUTPUT_FISHERFACES/fisherface' + '_reconstruction_' + str(i) + '.png'
#     image = plt.imread(path)
#     plt.subplot(4, 4, i + 1)
#     plt.imshow(image, plt.cm.gray)
#     plt.axis('off')
#     # plt.subplots_adjust(wspace = 0, hspace = 0)
# plt.show()