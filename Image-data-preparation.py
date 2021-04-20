def prepare_images(src, dst, img_size=224, bw=False):
    """It process project images and saves them to destination in .png format.
    Images are being resized and renamed. 
    
    :param src: path to folder containing raw images.
    :param dst: path to destination folder with processed images
    :param img_size: (optional) desired image sized. By default set to 224x224 pixels.
    :param bw: (optional) converting image to gray scale. By default image is converted to RGB.
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
    
    
def create_label_dict(src):
    """It gets list of labels based on your folder names in source directory, 
    assign numerical labels and returns a dictionary.
    
    :param src: path to folder containing raw images. """
    
    import os
    src = src.lstrip('\u202a')
    num_gen = (_ for _ in range(1, 101)) 
    
    label_dict = {species:next(num_gen)
            for species
            in os.listdir(src)}

    return label_dict
 

def create_dataset(directory, dst, label_dict, csv=False, npy=False):
    """It creates pandas data frame containing image names, images in form numpy arrays 
    and numerical labels.
    
    :param directory: path to general directory of a project
    :param dst: path to destination folder with processed images
    :param label_dict: dictionary containing numerical labels
    :param csv: (optional) saves dataset as csv to project directory. Off by default.
    :param npy: (optional) saves dataset in form of numpy (.npy) file to project directory. Off by default.
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
    
    return full_data


def remove_junk_channels(full_data):
    """For BW images only. Reduces redundant pixel data from 3 to 1 channel. Returns a list.
    
    :param full_data: Dataset in format of Pandas DataFrame."""
    
    X = []

    for im in full_data['arrays']:
        img = im.reshape(224,224,3)
        img = img[:,:,1].ravel()
        X.append(img)
        
    return X
