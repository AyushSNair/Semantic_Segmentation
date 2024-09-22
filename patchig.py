patch_size = 1000
def patched_data_load(folder_dir, patch_size):
    image_dataset = []
    # scaler = MinMaxScaler()
    random.seed(a=42)
    for images in tqdm(random.sample(os.listdir(folder_dir), 100)):
        image = cv2.imread(folder_dir+'/'+images, 1)
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        # SIZE_X = (image.shape[1]//patch_size)*patch_size #Nearest size divisible by our patch size
        # SIZE_Y = (image.shape[0]//patch_size)*patch_size #Nearest size divisible by our patch size
        # image = cv2.resize(image, (128, 128))
        image = Image.fromarray(image)
        # image = image.crop((0 ,0, SIZE_X, SIZE_Y))  #Crop from top left corner
        image = np.array(image)
        # print("Now patchifying image:", folder_dir+"/"+images)
        patches_img = patchify(image, (patch_size, patch_size, 3), step=patch_size)  #Step=256 for 256 patches means no overlap
        for i in range(patches_img.shape[0]):
            for j in range(patches_img.shape[1]):
                single_patch_img = patches_img[i,j,:,:]
                #Use minmaxscaler instead of just dividing by 255. 
                # single_patch_img = scaler.fit_transform(single_patch_img.reshape(-1, single_patch_img.shape[-1])).reshape(single_patch_img.shape)
                #single_patch_img = (single_patch_img.astype('float32')) / 255. 
                single_patch_img = single_patch_img[0] #Drop the extra unecessary dimension that patchify adds.   
                single_patch_img = cv2.resize(single_patch_img, (128, 128))
                image_dataset.append(single_patch_img)
    return image_dataset

patched_image_dataset = patched_data_load("../data/semantic_drone_dataset/training_set/images", patch_size)
patched_mask_dataset = patched_data_load("../data/semantic_drone_dataset/training_set/gt/semantic/label_images/", patch_size)

image_number = random.randint(0, len(patched_image_dataset))
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(patched_image_dataset[image_number])
plt.subplot(122)
plt.imshow(patched_mask_dataset[image_number])
plt.show()

def rgb_to_labels(img, mask_labels):
    label_seg = np.zeros(img.shape,dtype=np.uint8)
    for i in range(mask_labels.shape[0]):
        label_seg[np.all(img == list(mask_labels.iloc[i, [1,2,3]]), axis=-1)] = i
    label_seg = label_seg[:,:,0]  #Just take the first channel, no need for all 3 channels
    return label_seg

patched_labels = []
for i in tqdm(range(patched_mask_dataset.shape[0])):
    patched_label = rgb_to_labels(patched_mask_dataset[i], mask_labels)
    patched_labels.append(patched_label)
patched_labels = np.array(patched_labels)
patched_labels = np.expand_dims(patched_labels, axis=3)

print("Unique labels in label dataset are: ", np.unique(patched_labels))

image_number = random.randint(0, len(patched_image_dataset))
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(patched_image_dataset[image_number])
plt.subplot(122)
plt.imshow(patched_labels[image_number][:,:,0])
plt.show()

n_classes = len(np.unique(patched_labels))
patched_labels_cat = to_categorical(patched_labels, num_classes=n_classes)

X_train, X_test, y_train, y_test = train_test_split(patched_image_dataset, patched_labels_cat, test_size = 0.20, random_state = 42)