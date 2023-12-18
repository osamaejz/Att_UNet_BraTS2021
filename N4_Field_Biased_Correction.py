import nibabel as nib
import glob
import SimpleITK as sitk

# # Load the NIfTI image

t2_list = sorted(glob.glob('data/BraTS2021_train/*/*t2.nii.gz'))
t1ce_list = sorted(glob.glob('data/BraTS2021_train/*/*t1ce.nii.gz'))
flair_list = sorted(glob.glob('data/BraTS2021_train/*/*flair.nii.gz'))
mask_list = sorted(glob.glob('data/BraTS2021_train/*/*seg.nii.gz'))

#Each volume generates 18 64x64x64x4 sub-volumes. 
#Total 369 volumes = 6642 sub volumes

for img in range(1150,len(t2_list)): 
    
    inputImage_t2 = sitk.ReadImage(t2_list[img])
    inputImage_t1ce = sitk.ReadImage(t1ce_list[img])
    inputImage_flair = sitk.ReadImage(flair_list[img])
    
    numberOfIterations = 50
    numberOfFittingLevels = 5
    print(img)
    print("N4 bias correction runs.")
    maskImage = sitk.ReadImage(mask_list[img])
    maskImage = sitk.OtsuThreshold(maskImage,0,1,200)
    print("Skipping mask saving.. .")
    print("moving on.. .")

    corrector = sitk.N4BiasFieldCorrectionImageFilter();
    corrector.SetMaximumNumberOfIterations([int ( numberOfIterations)] * numberOfFittingLevels)
    
    inputImage_t2 = sitk.Cast(inputImage_t2,sitk.sitkFloat32)
    inputImage_t1ce = sitk.Cast(inputImage_t1ce,sitk.sitkFloat32)
    inputImage_flair = sitk.Cast(inputImage_flair,sitk.sitkFloat32)
        
    output_t2 = corrector.Execute(inputImage_t2,maskImage)
    output_t1ce = corrector.Execute(inputImage_t1ce,maskImage)
    output_flair = corrector.Execute(inputImage_flair,maskImage)

    
    print("Saving bias field corrected volume.. .")
    sitk.WriteImage(output_t2, t2_list[img])
    sitk.WriteImage(output_t1ce, t1ce_list[img])
    sitk.WriteImage(output_flair, flair_list[img])
    print("Finished N4 Bias Field Correction.....")
    
 