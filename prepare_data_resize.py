# 원본 이미지가 4k로 너무 크기때문에 resize를 수행함
import os
from pathlib import Path
import json
from tqdm import tqdm
from PIL import Image
from PIL import ImageFile # 추가
from utils.dataset import copy_folder


CWD = Path(__file__).parent


# 하단은 Configuration 값들
DEST_PATH = CWD / 'dataset' / 'farmsall'
min_resize = 1024    # 가로 또는 세로중 작은쪽의 크기가 min_resize 보다 큰 경우 resize 수행


# 대상 작물
SOURCE_SETS = [
            'strawberry_104',
            'strawberry_071',
            'strawberry_outsource',            
            'ood_02', 'ood_03','ood_05',
            'ood_06','ood_07','ood_08','ood_09','ood_10',
            'tomato_104',
]

SPLITS = [
            'train',
            'valid'
            ]

# 대상 이미지와 label 정보가 저장된 위치
PATHS_IMAGES = {'strawberry_104':
                    {'train': [ 'org/104.plant.disease.all/01.Data/1.Training/images/딸기/병해',
                                'org/104.plant.disease.all/01.Data/1.Training/images/딸기/생리장해',
                                #'org/104.plant.disease.all/01.Data/1.Training/images/딸기/작물보호제처리반응',
                                'org/104.plant.disease.all/01.Data/1.Training/images/딸기/정상'
                                ],
                    'valid': ['org/104.plant.disease.all/01.Data/2.Validation/images/딸기/병해',
                                'org/104.plant.disease.all/01.Data/2.Validation/images/딸기/생리장해',
                                #'org/104.plant.disease.all/01.Data/2.Validation/images/딸기/작물보호제처리반응',
                                'org/104.plant.disease.all/01.Data/2.Validation/images/딸기/정상'
                                ]
                    },
                'strawberry_071':
                    {
                        'train': ['org/071.farm.plant.disease/01.Data/1.Training/images/딸기/0.정상',
                                    'org/071.farm.plant.disease/01.Data/1.Training/images/딸기/1.질병',
                                    # 'org/071.farm.plant.disease/01.Data/1.Training/images/04.딸기/9.증강',
                        ],
                        'valid': ['org/071.farm.plant.disease/01.Data/2.Validation/images/딸기/0.정상',
                                    'org/071.farm.plant.disease/01.Data/2.Validation/images/딸기/1.질병',
                                    # 'org/071.farm.plant.disease/01.Data/2.Validation/images/04.딸기/9.증강',
                        ]
                    },
                'ood_02':
                    {   
                        'valid':['org/ood/images/02.normal','org/ood/images/02.disease']
                    },
                'ood_03':
                    {
                        'valid':['org/ood/images/03.normal','org/ood/images/03.disease',]
                    },
                'ood_05':
                    {
                        'valid':['org/ood/images/05.normal','org/ood/images/05.disease']
                    },
                'ood_06':
                    {
                        'valid':['org/ood/images/06.normal','org/ood/images/06.disease']
                    },
                'ood_07':
                    {
                        'valid':['org/ood/images/07.normal','org/ood/images/07.disease']
                    },
                'ood_08':
                    {
                        'valid':['org/ood/images/08.normal','org/ood/images/08.disease']
                    },
                'ood_09':
                    {
                        'valid':['org/ood/images/09.normal','org/ood/images/09.disease']
                    },
                'ood_10':
                    {
                        'valid':['org/ood/images/10.normal','org/ood/images/10.disease']
                    },
                # [06.13/20:15] folder tree 생성
                'strawberry_outsource':
                    {
                        'train': ['org/outsource/train/꽃곰팡이병',
                                    'org/outsource/train/시들음병',
                                    'org/outsource/train/잿빛곰팡이병',
                                    'org/outsource/train/점박이응애',
                                    'org/outsource/train/정상',
                                    'org/outsource/train/칼슘결핍',
                                    'org/outsource/train/탄저병',
                                    'org/outsource/train/흰가루병'
                        ],
                        'valid': ['org/outsource/valid/꽃곰팡이병',
                                    'org/outsource/valid/시들음병',
                                    'org/outsource/valid/잿빛곰팡이병',
                                    'org/outsource/valid/점박이응애',
                                    'org/outsource/valid/정상',
                                    'org/outsource/valid/칼슘결핍',
                                    'org/outsource/valid/탄저병',
                                    'org/outsource/valid/흰가루병'
                        ]
                    },
                'tomato_104':
                    {'train': [ 'org/104.plant.disease.all/01.Data/1.Training/images/토마토/병해',
                                'org/104.plant.disease.all/01.Data/1.Training/images/토마토/생리장해',
                                #'org/104.plant.disease.all/01.Data/1.Training/images/토마토/작물보호제처리반응',
                                'org/104.plant.disease.all/01.Data/1.Training/images/토마토/정상'
                                ],
                    'valid': ['org/104.plant.disease.all/01.Data/2.Validation/images/토마토/병해',
                                'org/104.plant.disease.all/01.Data/2.Validation/images/토마토/생리장해',
                                #'org/104.plant.disease.all/01.Data/2.Validation/images/토마토/작물보호제처리반응',
                                'org/104.plant.disease.all/01.Data/2.Validation/images/토마토/정상'
                                ]
                    },
}


PATHS_LABELS = {'strawberry_104':
                    {'train': [ 'org/104.plant.disease.all/01.Data/1.Training/labels/딸기/병해',
                                'org/104.plant.disease.all/01.Data/1.Training/labels/딸기/생리장해',
                                # 'org/104.plant.disease.all/01.Data/1.Training/labels/딸기/작물보호제처리반응',
                                'org/104.plant.disease.all/01.Data/1.Training/labels/딸기/정상',
                            ],
                    'valid': ['org/104.plant.disease.all/01.Data/2.Validation/labels/딸기/병해',
                                'org/104.plant.disease.all/01.Data/2.Validation/labels/딸기/생리장해',
                                #'org/104.plant.disease.all/01.Data/2.Validation/labels/딸기/작물보호제처리반응',
                                'org/104.plant.disease.all/01.Data/2.Validation/labels/딸기/정상',
                            ]
                    },
                'strawberry_071':
                    {'train': ['org/071.farm.plant.disease/01.Data/1.Training/labels/딸기/0.정상',
                                'org/071.farm.plant.disease/01.Data/1.Training/labels/딸기/1.질병',
                                # 'org/071.farm.plant.disease/01.Data/1.Training/labels/04.딸기/9.증강'
                            ],
                    'valid': ['org/071.farm.plant.disease/01.Data/2.Validation/labels/딸기/0.정상',
                                'org/071.farm.plant.disease/01.Data/2.Validation/labels/딸기/1.질병',
                                # 'org/071.farm.plant.disease/01.Data/1.Validation/labels/04.딸기/9.증강'
                            ]
                    },
                'ood_02':
                    {   
                        'valid':['org/ood/labels/02.normal','org/ood/labels/02.disease']
                    },
                'ood_03':
                    {
                        'valid':['org/ood/labels/03.normal','org/ood/labels/03.disease',]
                    },
                'ood_05':
                    {
                        'valid':['org/ood/labels/05.normal','org/ood/labels/05.disease']
                    },
                'ood_06':
                    {
                        'valid':['org/ood/labels/06.normal','org/ood/labels/06.disease']
                    },
                'ood_07':
                    {
                        'valid':['org/ood/labels/07.normal','org/ood/labels/07.disease']
                    },
                'ood_08':
                    {
                        'valid':['org/ood/labels/08.normal','org/ood/labels/08.disease']
                    },
                'ood_09':
                    {
                        'valid':['org/ood/labels/09.normal','org/ood/labels/09.disease']
                    },
                'ood_10':
                    {
                        'valid':['org/ood/labels/10.normal','org/ood/labels/10.disease']
                    },
                'tomato_104':
                    {'train': [ 'org/104.plant.disease.all/01.Data/1.Training/labels/토마토/병해',
                                'org/104.plant.disease.all/01.Data/1.Training/labels/토마토/생리장해',
                                # 'org/104.plant.disease.all/01.Data/1.Training/labels/토마토/작물보호제처리반응',
                                'org/104.plant.disease.all/01.Data/1.Training/labels/토마토/정상',
                            ],
                    'valid': ['org/104.plant.disease.all/01.Data/2.Validation/labels/토마토/병해',
                                'org/104.plant.disease.all/01.Data/2.Validation/labels/토마토/생리장해',
                                #'org/104.plant.disease.all/01.Data/2.Validation/labels/토마토/작물보호제처리반응',
                                'org/104.plant.disease.all/01.Data/2.Validation/labels/토마토/정상',
                            ]
                    },
}


# resize된 이미지가 저장될 위치
DEST_IMAGES = {'strawberry_104': {'train': DEST_PATH / 'train', 
                                    'valid': DEST_PATH / 'valid'}, 
                'strawberry_071': {'train': DEST_PATH / 'train', 
                                    'valid': DEST_PATH / 'valid'},
                # [05.08/16:20] folder path 추가
                'strawberry_outsource': {'train': DEST_PATH / 'train',
                                'valid': DEST_PATH / 'valid'},
                # [08.04] ood 추가
                'ood_02': {'valid': DEST_PATH / 'ood'/ 'pepper'},
                'ood_03': {'valid': DEST_PATH / 'ood'/ 'squash'},
                'ood_05': {'valid': DEST_PATH / 'ood'/ 'lettuce'},
                'ood_06': {'valid': DEST_PATH / 'ood' / 'watermelon'},
                'ood_07': {'valid': DEST_PATH / 'ood'/ 'zucchini'},
                'ood_08': {'valid': DEST_PATH / 'ood'/ 'zucchini2'},
                'ood_09': {'valid': DEST_PATH / 'ood'/ 'cucumber'},
                'ood_10': {'valid': DEST_PATH / 'ood'/ 'koreanmelon'},
                'tomato_104': {'train': DEST_PATH / 'ood' / 'tomato104' / 'train', 
                                    'valid': DEST_PATH / 'ood' / 'tomato104' / 'valid'},
                }


def resize(fullpath:Path, label:str, dest:Path, ratio:float=0.1, min_size:int=None):
    (dest / label).mkdir(parents=True, exist_ok=True)

    if fullpath.exists() is False:
        # 파일 확장자가 jpg로 일괄적이지 않음
        if fullpath.suffix == '.JPG':
            fullpath = fullpath.with_suffix('.jpg')
        elif fullpath.suffix == '.jpeg':
            fullpath = fullpath.with_suffix('.jpeg')
        else:
            fullpath = fullpath.with_suffix('.JPG')

    try:
        # truncated 오류 해결 -> 다운받다가 crop된 것들
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        img = Image.open(fullpath)
    except FileNotFoundError:
        return

    # 가로 또는 세로중 작은쪽의 크기가 min_resize 보다 큰 경우 resize 수행
    if min_size:
        if img.width > img.height and min_size < img.height:
            ratio = min_size / img.height
        elif img.width <= img.height and min_size < img.width:
            ratio = min_size / img.width
    
    if ratio:
        dest_width = int(img.width * ratio)
        dest_height = int(img.height * ratio)
        resized_img = img.resize((dest_width, dest_height))
    else:
        resized_img = img
        
    filename = fullpath.stem + fullpath.suffix.lower()
    dest_path = CWD / dest / label / filename
    resized_img.save(dest_path)


def resize_image_bbset(info:dict, path_image:Path, dest:Path, ratio:float=0.25, min_size:int=None):
    """데이터셋에 대한 resize"""
    if isinstance(dest, str): dest = Path(dest)
    if isinstance(path_image, str): path_image = Path(path_image)

    filename = info['description']['image']
    # 파일중 라벨폴더에는 존재하나 이미지는 존재하지 않는다면 pass 
    try:
        fullpath = path_image / filename
    except FileNotFoundError:
        print(filename+"파일은 이미지파일에 존재하지 않습니다")
        
    label = info['annotations']['disease']   # 질병 코드
    # 23.07.06
    map_labels = {  #외주데이터
                    0: "0", # "정상",
                    "c1": "c1", # "점박이응애",
                    "b1": "d1", #"꽃곰팡이병",
                    "b6": "p1", # 다량원소결핍(N, 질소)
                    "b7": "p1", # 다량원소결핍(P, 인)
                    "b8": "p1", # 다량원소결핍(K, 칼륨)
                    "a2": "d4", # 딸기흰가루병
                    "a1": "d2", # 딸기잿빛곰팡이병
                    # 2020: 071.시설작물 병해 데이터
                    7: "d2",
                    8: "d4",
                    # 2021: 104.식물병 통합유발 데이터
                    "00": "0", # "정상",
    }
    
    
    label = map_labels.get(label, label)    # if not found, bypass it
    resize(fullpath, str(label), dest, ratio, min_size)

def resize_image_ood(info:dict, path_image:Path, dest:Path, ratio:float=0.25, min_size:int=None):
    """ood 데이터셋에 대한 resize"""
    if isinstance(dest, str): dest = Path(dest)
    if isinstance(path_image, str): path_image = Path(path_image)

    filename = info['description']['image']
    # 파일중 라벨폴더에는 존재하나 이미지는 존재하지 않는다면 pass 
    try:
        fullpath = path_image / filename
    except FileNotFoundError:
        print(filename+"파일은 이미지파일에 존재하지 않습니다") 

    label = info['annotations']['disease']   # 질병 코드
    # 23.08.04
    map_labels = {  #ood
                    0: "0", # "정상",                    
                    3: "3", # 고추마일드모틀바이러스병
                    4: "4", # 고추점무늬병
                    5: "5", # 단호박노균병
                    6: "6", # 단호박흰가루병
                    9: "9", # 상추균핵병
                    10: "10", # 상추노균병
                    11: "11", # 수박탄저병
                    12: "12", # 수박흰가루병
                    13: "13", # 애호박점무늬병
                    14: "14", # 오이모자이크바이러스
                    15: "15", # 쥬키니호박 오이녹반모자이크바이러스
                    16: "16", # 참외노균병
                    17: "17", # 참외흰가루병
    }       
        
    label = map_labels.get(label, label)    # if not found, bypass it
    resize(fullpath, str(label), dest, ratio, min_size)

# json 데이터 없을때 클래스별 폴더구분 resize 함수
def resize_image_clfset(info:str, path_image:Path, dest:Path, ratio:float=0.25, min_size:int=None):
    if isinstance(dest, str): dest = Path(dest)
    if isinstance(path_image, str): path_image = Path(path_image)
    
    filename = info
    fullpath = path_image / filename
    label = filename.split('.')[0].split('_')[3]   # 질병 코드

    # Mapping x 폴더 생성 -> 병해 Class 별도 정리
    # Label mapping 작업
    map_labels = {
                    #외주데이터
                    0: "0", # "정상",
                    "0": "0",
                    "c1": "c1", # "해충피해(점박이응애)",
                    "b1": "d1", # "꽃곰팡이병",
                    "b6": "d3", # "시들음병",  -> "d3": "시들음병"
                    "b7": "p1", # "칼슘결핍",  -> "b9": "칼슘결핍"
                    "b8": "d5", # "탄저병", ->"d5": "탄저병"
                    "a2": "d4", # 딸기잿빛곰팡이병
                    "a1": "d2", # 딸기흰가루병
                    #2020 071.시설작물 병해 데이터
                    7: "d2", # 딸기잿빛곰팡이병
                    8: "d4", # 딸기흰가루병
                    #2021 104.식물병 통합유발 데이터
                    "00": "0" # 정상
    }
    label = map_labels.get(label, label)    # if not found, bypass it
    resize(fullpath, label, dest, ratio, min_size)


# Main 코드(strawberry resize) json 없는 데이터 파일명 활용
for fruit in SOURCE_SETS:
    print(fruit)
    if fruit == "strawberry_outsource":
        for ds in SPLITS:
            print('\t', ds)
            for path_images in PATHS_IMAGES[fruit][ds]:
                print('\t\t', path_images)
                abspath_images = CWD / path_images
                info = os.listdir(abspath_images)
                for infos in tqdm(info):
                    resize_image_clfset(infos, abspath_images, DEST_IMAGES[fruit][ds], None, min_resize)
    elif 'ood' in fruit:
        for ds in SPLITS:
            print('\t', ds)
            if ds in PATHS_LABELS[fruit]: 
                for path_labels, path_images in zip(PATHS_LABELS[fruit][ds], PATHS_IMAGES[fruit][ds]):
                    print('\t\t', path_labels)
                    abspath_images = CWD / path_images
                    abspath_labels = CWD / path_labels
                    for entry in tqdm(list(abspath_labels.iterdir())):
                        if entry.is_file():
                            with entry.open() as fp:
                                info = json.load(fp)
                                resize_image_ood(info, abspath_images, DEST_IMAGES[fruit][ds], None, min_resize)        
    else:
        for ds in SPLITS:
            print('\t', ds)
            for path_labels, path_images in zip(PATHS_LABELS[fruit][ds], PATHS_IMAGES[fruit][ds]):
                print('\t\t', path_labels)
                abspath_images = CWD / path_images
                abspath_labels = CWD / path_labels
                for entry in tqdm(list(abspath_labels.iterdir())):
                    if entry.is_file():
                        with entry.open() as fp:
                            info = json.load(fp)
                            resize_image_bbset(info, abspath_images, DEST_IMAGES[fruit][ds], None, min_resize)
