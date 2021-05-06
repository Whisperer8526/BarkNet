def process_images(src, dst, img_size=224, bw=False):
    """
    Process project images and saves them to destination in .png format.
       
    Arguments:
        src: path to folder containing raw images.
        dst: path to destination folder with processed images
        img_size: (optional) desired image sized. By default set to 224x224 pixels.
        bw: (optional) converting images to black and white (3 channels).
    Returns:
        Doesn't return any object. Saves processed images to destination folder.
    """
    import os, cv2
    src = src.lstrip('\u202a')
    dst = dst.lstrip('\u202a')
    
    for species in os.listdir(src):  
        path = os.path.join(src, species)
        num_gen = (_ for _ in range(1, 100001))
    
        for img in os.listdir(path):
            try:
                image = cv2.imread(os.path.join(path, img))                 # reading image
                resized_image = cv2.resize(image, (img_size, img_size))     # resizing image
                image_label = f"{next(num_gen)}-{species}.png"              # renaming image
                
                if bw == True:
                    bw_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY) # converting to BW
                    cv2.imwrite(os.path.join(dst, image_label), bw_image)      # saving image
                    print(f"\nImage {image_label} saved successfully.", end = "\r")
                else:
                    rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB) # converting to RGB
                    cv2.imwrite(os.path.join(dst, image_label), rgb_image)     # saving image
                    print(f"\nImage {image_label} saved successfully.", end = "\r")  
            
            except Exception:
                print(f"{img} : conversion error")

                
def binarize_images(src, dst, img_size=224):
    """
    Pipeline processing images to binary contrast. Includes renaming and resizing. 
    
    Arguments:
        src: path to folder containing raw images.
        dst: path to destination folder with processed images
        img_size: (optional) desired image sized. By default set to 224x224 pixels.
    Returns:
        Doesn't return any object. Saves processed images to destination folder in .png format
    """
    import os, cv2
    src = src.lstrip('\u202a')
    dst = dst.lstrip('\u202a')
    clahe = cv2.createCLAHE(clipLimit=40)
    th=80
    max_val=255
    
    for species in os.listdir(src):  
        path = os.path.join(src, species)
        num_gen = (_ for _ in range(1, 100001))
    
        for img in os.listdir(path):
            try:
                image = cv2.imread(os.path.join(path, img))                 # reading 
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)             # conversion to BW
                image_eqhist = cv2.equalizeHist(image)                      # histogram equalization
                image_clahe = clahe.apply(image_eqhist)
                ret, image_tresh1 = cv2.threshold(image_clahe, th, max_val,  cv2.THRESH_OTSU)
                image_blur = cv2.medianBlur(image_tresh1,3)                 # applying blur
                ret, image_tresh2 = cv2.threshold(image_blur,0, 255, 
                                                  cv2.THRESH_BINARY+cv2.THRESH_OTSU) # image binarizatiom
                resized_image = cv2.resize(image_tresh2, (img_size, img_size))# resizing 
                final_image = cv2.medianBlur(resized_image,3)                 # applying blur
                labelled_image = f"{next(num_gen)}-{species}-BC.png"          # renaming 
                cv2.imwrite(os.path.join(dst, labelled_image), final_image)   # saving 
                print(f"\nImage {labelled_image} saved successfully.", end = "\r")

            except Exception:
                print(f"{img} : conversion error")
    
def create_label_dict(src):
    """
    It gets list of labels based on your folder names in source directory, 
    assign numerical labels and returns a dictionary.
    
    Arguments:
        src: path to folder containing raw images. 
    Returns:
        Dictionary with numeric labels
    """
    
    import os
    src = src.lstrip('\u202a')
    num_gen = (_ for _ in range(0, 101)) 
    
    label_dict = {species:next(num_gen)
            for species
            in os.listdir(src)}

    return label_dict
 

def create_dataset(directory, dst, label_dict, csv=False, npy=False):
    """
    Creates pandas data frame containing image names, images in form numpy arrays 
    and numerical labels.
    
    Arguments: 
        directory: path to general directory of a project
        dst: path to destination folder with processed images
        label_dict: dictionary containing numerical labels
        csv: (optional) saves dataset as csv to project directory. Off by default.
        npy: (optional) saves dataset in form of numpy (.npy) file to project directory. Off by default.
    Returns:
        Pandas DataFrame containing dataset. 
    """
    
    import os, cv2, pandas, numpy
    directory = directory.lstrip('\u202a')
    dst = dst.lstrip('\u202a')

    image_names = [img for img in os.listdir(dst)]
    image_arrays = []
    image_labels =[]
       
    for img in os.listdir(dst):                        
        num_img = cv2.imread(os.path.join(dst, img))   #converting image into numpy arrays
        image_arrays.append(num_img.flatten())         #and flattening them for scaling purposes
        for species in label_dict:                     #labelling images
            if species in img:
                image_labels.append(label_dict[species])
                
        data = {'names' : image_names,
            'arrays' : image_arrays,
            'labels' : image_labels}
    

    full_data = pandas.DataFrame(data)                 #saving dataset to pandas data frame
    
    if csv == True:
        full_data.to_csv(os.path.join(directory, "project_data.csv"))
    
    if npy == True:
        numpy.save(os.path.join(directory, "image_arrays_rgb.npy"), image_arrays)
        numpy.save(os.path.join(directory, "image_labels.npy"), image_labels)
    
    return dataset


def remove_junk_channels(dataset):
    """
    For BW images only. Reduces redundant pixel data from 3 to 1 channel. Final list contains 1D numpy arrays. 
    
    Arguments:
        data: Dataset in format of Pandas DataFrame.
    Returns: 
        List with BW image data.
    """
    bw_images = []

    for im in dataset['arrays']:
        img = im.reshape(224,224,3)
        img = img[:,:,1].ravel()
        bw_images.append(img)
        
    return bw_images
