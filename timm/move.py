import shutil
import os
kinds1 = ['All','Armored','Ground','Plane','Tank','Truck']
fusions = ['Input','Decision','Feature']
kinds2 = ['Ship','Harbor']
yuan1 = ['RGB','INF']
yuan2 = ['RGB','SAR','PAN']
# for kind in kinds1:
#     for fusion in fusions:
#         if not os.path.exists(f'quanzhong/{kind}'):
#             os.mkdir(f'quanzhong/{kind}')
#         # if not os.path.exists(f'quanzhong/{kind}/{fusion}_Fusion'):
#         #     os.mkdir(f'quanzhong/{kind}/{fusion}_Fusion')
#         shutil.copy(f'work_dirs/resnet50_{kind}/{fusion}_Fusion/model_100.pth',f'quanzhong/{kind}/{fusion}_Fusion.pth')
# for kind in kinds2:
#     for fusion in fusions:
#         if not os.path.exists(f'quanzhong/{kind}'):
#             os.mkdir(f'quanzhong/{kind}')
#         # if not os.path.exists(f'quanzhong/{kind}/{fusion}_Fusion'):
#         #     os.mkdir(f'quanzhong/{kind}/{fusion}_Fusion')
#         shutil.copy(f'work_dirs/resnet50_{kind}/{fusion}_Fusion/model_100.pth',f'quanzhong/{kind}/{fusion}_Fusion.pth')
# for kind in kinds1:
#     for yuan in yuan1:
#         if not os.path.exists(f'quanzhong/{kind}'):
#             os.mkdir(f'quanzhong/{kind}')
#         # if not os.path.exists(f'quanzhong/{kind}/No_Fusion_{yuan}'):
#         #     os.mkdir(f'quanzhong/{kind}/No_Fusion_{yuan}')
#         shutil.copy(f'work_dirs/resnet50_{kind}/No_Fusion_{yuan}/model_100.pth',f'quanzhong/{kind}/No_Fusion_{yuan}.pth')
# for kind in kinds2:
#     for yuan in yuan2:
#         if not os.path.exists(f'quanzhong/{kind}'):
#             os.mkdir(f'quanzhong/{kind}')
#         # if not os.path.exists(f'quanzhong/{kind}/No_Fusion_{yuan}'):
#         #     os.mkdir(f'quanzhong/{kind}/No_Fusion_{yuan}')
#         shutil.copy(f'work_dirs/resnet50_{kind}/No_Fusion_{yuan}/model_100.pth',f'quanzhong/{kind}/No_Fusion_{yuan}.pth')