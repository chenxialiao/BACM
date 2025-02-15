# data_config = {
#     "CASIA": {
#         "image_directory": "/home/liao/TEST/OSFI-by-FineTuning/CASIA_clean/cropped_images"
#     },
#     "IJBC":{
#         "image_directory" : ""
#     },
# }
#
# encoder_config = {
#     "VGG19": "/home/liao/TEST/OSFI-by-FineTuning/VGG19_CosFace.chkpt",
#     "Res50": "/home/liao/TEST/OSFI-by-FineTuning/ResIR50_CosFace.chkpt",
# }


import  os

data_config  =  {
     "CASIA":  {
         "image_directory":  "/home/liao/TEST/OSFI-by-FineTuning/CASIA_clean/cropped_images",
         "known_list_path":  "/home/liao/TEST/OSFI-by-FineTuning/CASIA_clean/CASIA_known_list.pkl",
         "unknown_list_path":  "/home/liao/TEST/OSFI-by-FineTuning/CASIA_clean/CASIA_unknown_list.pkl",
     },
    "IJBC": {
        "image_directory": "/home/liao/TEST/IJB-C/cropped_images",
        "ijbc_t_m": "/home/liao/TEST/OSFI-by-FineTuning/IJB-C/ijbc_face_tid_mid.txt",
        "ijbc_5pts": "/home/liao/TEST/OSFI-by-FineTuning/IJB-C/ijbc_name_5pts_score.txt",
        "ijbc_gallery_1": "/home/liao/TEST/OSFI-by-FineTuning/IJB-C/ijbc_1N_gallery_G1.csv",
        "ijbc_gallery_2": "/home/liao/TEST/OSFI-by-FineTuning/IJB-C/ijbc_1N_gallery_G2.csv",
        "ijbc_probe": "/home/liao/TEST/OSFI-by-FineTuning/IJB-C/ijbc_1N_probe_mixed.csv",
        "processed_img_root": "/home/liao/TEST/IJB-C/cropped_images",  # 处理后图片存放路径
        "plk_file_root": "/home/liao/TEST/OSFI-by-FineTuning/IJB-C",  # 数据集划分成Gallery和Probe后存入pkl文件的路径
        "img_root_loose_crop": "/home/liao/TEST/ijb/IJBC/loose_crop",  # 未处理前图片存放路径
    },
}

encoder_config  =  {
     "VGG19":  "/home/liao/TEST/OSFI-by-FineTuning/VGG19_CosFace.chkpt",
     "Res50":  "/home/liao/TEST/OSFI-by-FineTuning/ResIR50_CosFace.chkpt",
}



