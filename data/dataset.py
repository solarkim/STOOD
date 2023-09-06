from typing import Optional, Callable, Any
from pathlib import Path
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader


class CropDisease(ImageFolder):
    LABELS = {"0": "정상",
                "c1": "점박이응애",
                "d1": "꽃곰팡이병",
                "d2": "흰가루병",
                "d3": "시들음병",
                "d4": "잿빛곰팡이병",
                "d5": "탄저병",
                "p1": "다량원소결핍"    # b6: (N,질소), b7: (P,인), b8: (K,칼륨)
            }  # folder명 -> 표시될 이름

    def __init__(self, root: str, kind: str = 'train',
                transform: Optional[Callable] = None,
                target_transform: Optional[Callable] = None,
                loader: Callable[[str], Any] = default_loader,
                is_valid_file: Optional[Callable[[str], bool]] = None):
        if kind in ['train', 'valid', 'test']:
            if kind == 'test':
                kind = 'valid'
            folder = Path(root) / kind
        else:
            folder = root
        super().__init__(folder, transform, target_transform, loader, is_valid_file)

    @property
    def labels(self) -> list:
        return [self.LABELS.get(label) for label in self.classes]



class PapperDisease(CropDisease):
    LABELS = {"0": "pepperG",
                "3": "pepper03",   # 고추마일드모틀바이러스
                # "4": "pepper04",   # 고추점무늬병
            }  # folder명 -> 표시될 이름

    def __init__(self, root: str, kind: str = 'train',
                transform: Optional[Callable] = None,
                target_transform: Optional[Callable] = None,
                loader: Callable[[str], Any] = default_loader,
                is_valid_file: Optional[Callable[[str], bool]] = None):
        super().__init__(root / 'pepper', kind, transform,
                        target_transform, loader, is_valid_file)


class SquashDisease(CropDisease):
    LABELS = {"0": "squashG",
                "5": "squash05",   # 단호박점무늬병
                "6": "squash06",   # 단호박흰가루병
            }  # folder명 -> 표시될 이름

    def __init__(self, root: str, kind: str = 'train',
                transform: Optional[Callable] = None,
                target_transform: Optional[Callable] = None,
                loader: Callable[[str], Any] = default_loader,
                is_valid_file: Optional[Callable[[str], bool]] = None):
        super().__init__(root / 'squash', kind, transform,
                        target_transform, loader, is_valid_file)


class LettuceDisease(CropDisease):
    LABELS = {"0": "lettuceG",
                "9": "lettuce09",   # 상추균핵병
                "10": "lettuce10",   # 상추노균병
            }  # folder명 -> 표시될 이름

    def __init__(self, root: str, kind: str = 'train',
                transform: Optional[Callable] = None,
                target_transform: Optional[Callable] = None,
                loader: Callable[[str], Any] = default_loader,
                is_valid_file: Optional[Callable[[str], bool]] = None):
        super().__init__(root / 'lettuce', kind, transform,
                        target_transform, loader, is_valid_file)


class WaterMelonDisease(CropDisease):
    LABELS = {"0": "wmelonG",
                "16": "wmelon16",   # 수박탄저병
                "17": "wmelon17",   # 수박흰가루병
            }  # folder명 -> 표시될 이름

    def __init__(self, root: str, kind: str = 'train',
                transform: Optional[Callable] = None,
                target_transform: Optional[Callable] = None,
                loader: Callable[[str], Any] = default_loader,
                is_valid_file: Optional[Callable[[str], bool]] = None):
        super().__init__(root / 'watermelon', kind, transform,
                        target_transform, loader, is_valid_file)


class ZucchiniDisease(CropDisease):
    LABELS = {"0": "zucchiG",
                "13": "zucchi13",   # 애호박점무늬병
            }  # folder명 -> 표시될 이름

    def __init__(self, root: str, kind: str = 'train',
                transform: Optional[Callable] = None,
                target_transform: Optional[Callable] = None,
                loader: Callable[[str], Any] = default_loader,
                is_valid_file: Optional[Callable[[str], bool]] = None):
        super().__init__(root / 'zucchini', kind, transform,
                        target_transform, loader, is_valid_file)


class Zucchini2Disease(CropDisease):
    LABELS = {"0": "zucchi2G",
                "15": "zucchi215",   # 오이녹반모자이크바이러스
            }  # folder명 -> 표시될 이름

    def __init__(self, root: str, kind: str = 'train',
                transform: Optional[Callable] = None,
                target_transform: Optional[Callable] = None,
                loader: Callable[[str], Any] = default_loader,
                is_valid_file: Optional[Callable[[str], bool]] = None):
        super().__init__(root / 'zucchini2', kind, transform,
                        target_transform, loader, is_valid_file)


class CucumberDisease(CropDisease):
    LABELS = {"0": "cucumbG",
                "14": "cucumb14",   # 오이모자이크바이러스
            }  # folder명 -> 표시될 이름

    def __init__(self, root: str, kind: str = 'train',
                transform: Optional[Callable] = None,
                target_transform: Optional[Callable] = None,
                loader: Callable[[str], Any] = default_loader,
                is_valid_file: Optional[Callable[[str], bool]] = None):
        super().__init__(root / 'cucumber', kind, transform,
                        target_transform, loader, is_valid_file)


class KoreanMelonDisease(CropDisease):
    LABELS = {"0": "kmelonG",
                "16": "kmelon16",   # 참외노균병
                "17": "wmelon17",   # 참외흰가루병
            }  # folder명 -> 표시될 이름

    def __init__(self, root: str, kind: str = 'train',
                transform: Optional[Callable] = None,
                target_transform: Optional[Callable] = None,
                loader: Callable[[str], Any] = default_loader,
                is_valid_file: Optional[Callable[[str], bool]] = None):
        super().__init__(root / 'koreanmelon', kind, transform,
                        target_transform, loader, is_valid_file)


class Tomato104(CropDisease):
    LABELS = {"0": "0",
                "a5": "a5",
                "a6": "a6",
                "b2": "b2",
                "b3": "b3",
                "p1": "p1",
            }  # folder명 -> 표시될 이름
