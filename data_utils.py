"""Dataset loading and client partitioning (local-first, no forced downloads).

RGB datasets are resized to config.image_size (default 32) so the existing
SimpleCNN is reused unchanged. Folder datasets must be laid out as
<root>/<domain>/<class>/*.{jpg,png,...}; a flat <root>/<class>/* layout is
also accepted (treated as a single domain).
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, transforms

Image.MAX_IMAGE_PIXELS = None

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

PACS_DOMAINS = ["art_painting", "cartoon", "photo", "sketch"]
OFFICE_DOMAINS = ["amazon", "caltech", "dslr", "webcam"]
DOMAINNET_DOMAINS = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".ppm", ".tif", ".tiff"}

# directory-name aliases so capitalisation variants resolve to one domain
DOMAIN_ALIASES = {"caltech10": "caltech", "caltech": "caltech",
                  "artpainting": "art_painting", "art painting": "art_painting"}

MANUAL_INFO = {
    "CIFAR10": ("https://cave.cs.toronto.edu/kriz/cifar-10-python.tar.gz",
                "cifar-10-python.tar.gz", "cifar-10-batches-py"),
    "CIFAR100": ("https://cave.cs.toronto.edu/kriz/cifar-100-python.tar.gz",
                 "cifar-100-python.tar.gz", "cifar-100-python"),
}


# ----------------------------------------------------------------------
# Containers / helpers
# ----------------------------------------------------------------------
class ImageListDataset(Dataset):
    def __init__(self, samples, classes, transform, domains=None, domain_names=None):
        self.samples = [(str(p), int(y)) for p, y in samples]
        self.classes = list(classes)
        self.targets = [int(y) for _, y in self.samples]
        self.transform = transform
        self.domains = [int(d) for d in domains] if domains is not None else None
        self.domain_names = list(domain_names) if domain_names else None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label


class FilteredDataset(Dataset):
    def __init__(self, base, indices):
        self.base = base
        self.indices = [int(i) for i in indices]
        t = np.asarray(labels_from_dataset(base), dtype=np.int64)
        self.targets = t[self.indices].tolist()
        self.classes = getattr(base, "classes", None)
        base_dom = getattr(base, "domains", None)
        self.domains = (np.asarray(base_dom)[self.indices].tolist()
                        if base_dom is not None else None)
        self.domain_names = getattr(base, "domain_names", None)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        return self.base[self.indices[i]]


def labels_from_dataset(dataset):
    if hasattr(dataset, "targets"):
        return np.asarray(dataset.targets, dtype=np.int64)
    if hasattr(dataset, "labels"):
        return np.asarray(dataset.labels, dtype=np.int64)
    return np.asarray([int(dataset[i][1]) for i in range(len(dataset))], dtype=np.int64)


def train_class_counts(dataset, num_classes):
    t = labels_from_dataset(dataset)
    return np.bincount(t, minlength=num_classes).astype(np.int64)


def _rgb_transform(image_size):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def _canon(name):
    key = name.strip().lower().replace(" ", "_")
    return DOMAIN_ALIASES.get(key, key)


def _has_images(d):
    return any(p.suffix.lower() in IMAGE_EXTS for p in d.iterdir() if p.is_file())


def _visible_dirs(path):
    return sorted([d for d in path.iterdir()
                   if d.is_dir() and not d.name.startswith(".")])


# ----------------------------------------------------------------------
# torchvision (local-first)
# ----------------------------------------------------------------------
def _manual_message(name, root):
    url, archive, folder = MANUAL_INFO.get(name, ("<url>", "<archive>", "<folder>"))
    return (f"{name} not present locally and download disabled/failed.\n"
            f"  cd {root}\n  wget -c {url}\n  tar -xzf {archive}\n"
            f"Then verify {root}/{folder} exists.")


def _load_tv(cls, name, root, train, transform, allow_download):
    try:
        return cls(root=str(root), train=train, download=False, transform=transform)
    except Exception:
        pass
    if not allow_download:
        raise FileNotFoundError(_manual_message(name, root))
    try:
        return cls(root=str(root), train=train, download=True, transform=transform)
    except Exception as exc:
        raise FileNotFoundError(_manual_message(name, root) + f"\n{exc}") from exc


# ----------------------------------------------------------------------
# Folder scanning
# ----------------------------------------------------------------------
def _resolve_root(root, extra=()):
    root = Path(root)
    for cand in (root,) + tuple(Path(c) for c in extra):
        if cand.is_dir():
            return cand
    raise FileNotFoundError(f"Dataset root not found: {root}")


def _scan_domain_class(root, allowed_domains, num_classes, max_per_class, seed):
    root = Path(root)
    if (root / "images").is_dir():
        root = root / "images"

    candidates = _visible_dirs(root)
    # keep only dirs that actually hold images (skips .git, README-only dirs,
    # and the leftover HF `data/` parquet dir)
    domain_dirs = [d for d in candidates if _has_images(d) or
                   any(_has_images(c) for c in _visible_dirs(d))]
    if not domain_dirs:
        raise FileNotFoundError(
            f"No domain folders containing images under {root}. "
            f"Found: {[d.name for d in candidates]}")

    if allowed_domains:
        wanted = {_canon(d) for d in allowed_domains}
        filtered = [d for d in domain_dirs if _canon(d.name) in wanted]
        if filtered:                      # fall back to all if filter misses
            domain_dirs = filtered

    # classes = union of class dirs across domains; a domain whose subdirs hold
    # images directly is treated as one implicit class (flat layout)
    class_names = sorted({c.name for d in domain_dirs for c in _visible_dirs(d)
                          if _has_images(c) or any(_has_images(g) for g in _visible_dirs(c))})
    if not class_names:
        raise FileNotFoundError(f"No class folders under {root}")

    rng = np.random.default_rng(seed)
    if num_classes is not None and num_classes < len(class_names):
        sel = np.sort(rng.choice(len(class_names), size=num_classes, replace=False))
        class_names = [class_names[i] for i in sel]
    cls_to_idx = {c: i for i, c in enumerate(class_names)}

    samples, domains = [], []
    for di, ddir in enumerate(domain_dirs):
        for cname in class_names:
            cdir = ddir / cname
            if not cdir.is_dir():
                continue
            files = sorted([p for p in cdir.iterdir()
                            if p.is_file() and p.suffix.lower() in IMAGE_EXTS])
            if not files:
                continue
            if max_per_class is not None and len(files) > max_per_class:
                pick = np.sort(rng.choice(len(files), size=max_per_class, replace=False))
                files = [files[i] for i in pick]
            for p in files:
                samples.append((p, cls_to_idx[cname]))
                domains.append(di)

    if not samples:
        raise FileNotFoundError(f"No images found under {root}")
    return samples, class_names, domains, [d.name for d in domain_dirs]


def _stratified_split(samples, domains, test_fraction, seed):
    rng = np.random.default_rng(seed)
    groups = defaultdict(list)
    for i, ((_, y), d) in enumerate(zip(samples, domains)):
        groups[(d, y)].append(i)
    tr, te = [], []
    for key in sorted(groups.keys()):
        idx = np.asarray(groups[key])
        rng.shuffle(idx)
        n_test = max(1, int(round(len(idx) * test_fraction))) if len(idx) > 1 else 0
        te.extend(idx[:n_test].tolist())
        tr.extend(idx[n_test:].tolist())
    return sorted(tr), sorted(te)


def _folder_dataset(root, allowed_domains, cfg, extra=()):
    root = _resolve_root(root, extra)
    samples, classes, domains, dnames = _scan_domain_class(
        root, allowed_domains,
        getattr(cfg, "subset_num_classes", None),
        getattr(cfg, "subset_max_per_class", None),
        getattr(cfg, "subset_seed", 0))
    tr, te = _stratified_split(samples, domains,
                               getattr(cfg, "subset_test_fraction", 0.2),
                               getattr(cfg, "subset_seed", 0))
    tf = _rgb_transform(getattr(cfg, "image_size", 32))
    train = ImageListDataset([samples[i] for i in tr], classes, tf,
                             [domains[i] for i in tr], dnames)
    test = ImageListDataset([samples[i] for i in te], classes, tf,
                            [domains[i] for i in te], dnames)
    return train, test


def _inat_subset(root, cfg):
    root = Path(root)
    ann_path = root / "train2018.json"
    n_cls = getattr(cfg, "inat_num_classes", 100)
    cap = getattr(cfg, "inat_max_per_class", 200)
    seed = getattr(cfg, "subset_seed", 0)
    rng = np.random.default_rng(seed)

    if ann_path.is_file():
        with ann_path.open("r", encoding="utf-8") as fp:
            ann = json.load(fp)
        cats = {c["id"]: c for c in ann["categories"]}
        imgs = {im["id"]: im for im in ann["images"]}
        by_cat = defaultdict(list)
        for a in ann["annotations"]:
            by_cat[a["category_id"]].append(a["image_id"])
        ordered = sorted(by_cat.keys(), key=lambda c: len(by_cat[c]))
        if n_cls is not None and n_cls < len(ordered):
            if getattr(cfg, "inat_preserve_longtail", True):
                pick = np.linspace(0, len(ordered) - 1, num=n_cls).round().astype(int)
                chosen = [ordered[i] for i in sorted(set(pick.tolist()))]
            else:
                chosen = [ordered[i] for i in
                          np.sort(rng.choice(len(ordered), n_cls, replace=False))]
        else:
            chosen = ordered
        samples, classes = [], []
        for new_id, cid in enumerate(chosen):
            classes.append(cats[cid].get("name", str(cid)))
            ids = sorted(set(by_cat[cid]))
            if cap is not None and len(ids) > cap:
                ids = [ids[i] for i in np.sort(rng.choice(len(ids), cap, replace=False))]
            for iid in ids:
                rel = imgs[iid]["file_name"]
                p = root / rel
                if not p.is_file():
                    p = root / "train_val2018" / rel
                if p.is_file():
                    samples.append((p, new_id))
        if not samples:
            raise FileNotFoundError(f"iNat images not found under {root}")
        domains, dnames = [0] * len(samples), ["inat2018"]
    else:
        root = _resolve_root(root)
        samples, classes, domains, dnames = _scan_domain_class(root, None, n_cls, cap, seed)

    tr, te = _stratified_split(samples, domains,
                               getattr(cfg, "subset_test_fraction", 0.2), seed)
    tf = _rgb_transform(getattr(cfg, "image_size", 32))
    train = ImageListDataset([samples[i] for i in tr], classes, tf,
                             [domains[i] for i in tr], dnames)
    test = ImageListDataset([samples[i] for i in te], classes, tf,
                            [domains[i] for i in te], dnames)
    return train, test


class _DefaultCfg:
    data_root = "./data"
    download = False
    image_size = 32
    subset_num_classes = 20
    subset_max_per_class = 150
    subset_test_fraction = 0.2
    subset_seed = 3042
    domainnet_domains = None
    inat_num_classes = 100
    inat_max_per_class = 200
    inat_preserve_longtail = True
    longtail_imbalance = None


def get_dataset(name, config=None):
    """Return (train_set, test_set, meta). meta is ALWAYS assigned."""
    cfg = config if config is not None else _DefaultCfg()
    root = Path(getattr(cfg, "data_root", "./data"))
    allow_dl = bool(getattr(cfg, "download", False))
    size = int(getattr(cfg, "image_size", 32))
    key = name.upper()

    if key in ("CIFAR10", "CIFAR100"):
        tf = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
        cls = datasets.CIFAR10 if key == "CIFAR10" else datasets.CIFAR100
        train = _load_tv(cls, key, root, True, tf, allow_dl)
        test = _load_tv(cls, key, root, False, tf, allow_dl)
        n_cls, in_ch, dom = (10, 3, False) if key == "CIFAR10" else (100, 3, False)
        class_names = list(train.classes)

    elif key == "MNIST":
        tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        train = _load_tv(datasets.MNIST, key, root, True, tf, allow_dl)
        test = _load_tv(datasets.MNIST, key, root, False, tf, allow_dl)
        size, n_cls, in_ch, dom = 28, 10, 1, False
        class_names = [str(i) for i in range(10)]

    elif key == "PACS":
        train, test = _folder_dataset(root / "PACS", PACS_DOMAINS, cfg)
        n_cls, in_ch, dom = len(train.classes), 3, True
        class_names = train.classes

    elif key in ("OFFICE_CALTECH10", "OFFICECALTECH10", "OFFICE_CALTECH"):
        train, test = _folder_dataset(
            root / "Office-Caltech10", OFFICE_DOMAINS, cfg,
            extra=(root / "office_caltech_10", root / "OfficeCaltech10",
                   root / "office_caltech"))
        n_cls, in_ch, dom = len(train.classes), 3, True
        class_names = train.classes

    elif key in ("DOMAINNET_SUBSET", "DOMAINNET"):
        train, test = _folder_dataset(
            root / "DomainNet",
            getattr(cfg, "domainnet_domains", None) or DOMAINNET_DOMAINS, cfg,
            extra=(root / "domainnet",))
        n_cls, in_ch, dom = len(train.classes), 3, True
        class_names = train.classes

    elif key in ("INAT2018_SUBSET", "INAT2018", "INAT"):
        train, test = _inat_subset(root / "inat2018", cfg)
        n_cls, in_ch, dom = len(train.classes), 3, False
        class_names = train.classes

    else:
        raise ValueError(f"Dataset {name} not supported.")

    imb = getattr(cfg, "longtail_imbalance", None)
    if imb:
        train = apply_longtail(train, float(imb), getattr(cfg, "subset_seed", 0))

    meta = dict(num_classes=int(n_cls), image_size=int(size), in_channels=int(in_ch),
                class_names=class_names, has_domains=bool(dom),
                train_size=len(train), test_size=len(test), dataset=key)
    return train, test, meta


# ----------------------------------------------------------------------
# Long-tail shaping
# ----------------------------------------------------------------------
def apply_longtail(dataset, imbalance, seed=0):
    targets = labels_from_dataset(dataset)
    classes = np.unique(targets)
    n_max = int(np.bincount(targets).max())
    rng = np.random.default_rng(seed)
    keep = []
    for k, c in enumerate(classes):
        idx = np.where(targets == c)[0]
        rng.shuffle(idx)
        frac = (1.0 / imbalance) ** (k / max(len(classes) - 1, 1))
        n_c = max(1, int(round(n_max * frac)))
        keep.extend(idx[:min(n_c, len(idx))].tolist())
    return FilteredDataset(dataset, sorted(keep))


# ----------------------------------------------------------------------
# Partitioning
# ----------------------------------------------------------------------
def dirichlet_partition(dataset, num_clients, alpha):
    targets = labels_from_dataset(dataset)
    client_indices = [[] for _ in range(num_clients)]
    for c in np.unique(targets):
        idx = np.where(targets == c)[0]
        np.random.shuffle(idx)
        props = np.random.dirichlet(np.repeat(alpha, num_clients))
        cuts = (np.cumsum(props) * len(idx)).astype(int)[:-1]
        for i, split in enumerate(np.split(idx, cuts)):
            client_indices[i].extend(split.tolist())
    return client_indices


def _assign_clients_to_domains(num_domains, num_clients):
    groups = [[] for _ in range(num_domains)]
    for cid in range(num_clients):
        groups[cid % num_domains].append(cid)
    return [g for g in groups if g]


def domain_partition(dataset, num_clients):
    dom = getattr(dataset, "domains", None)
    if dom is None:
        raise ValueError("domain_partition requires `.domains`.")
    dom = np.asarray(dom)
    uniq = np.unique(dom)
    groups = _assign_clients_to_domains(len(uniq), num_clients)
    out = [[] for _ in range(num_clients)]
    for d, cids in zip(uniq, groups):
        idx = np.where(dom == d)[0]
        np.random.shuffle(idx)
        for part, cid in zip(np.array_split(idx, len(cids)), cids):
            out[cid].extend(part.tolist())
    return out


def domain_dirichlet_partition(dataset, num_clients, alpha):
    dom = getattr(dataset, "domains", None)
    if dom is None:
        return dirichlet_partition(dataset, num_clients, alpha)
    dom = np.asarray(dom)
    targets = labels_from_dataset(dataset)
    uniq = np.unique(dom)
    groups = _assign_clients_to_domains(len(uniq), num_clients)
    out = [[] for _ in range(num_clients)]
    for d, cids in zip(uniq, groups):
        pool = np.where(dom == d)[0]
        for c in np.unique(targets[pool]):
            idx = pool[targets[pool] == c]
            np.random.shuffle(idx)
            props = np.random.dirichlet(np.repeat(alpha, len(cids)))
            cuts = (np.cumsum(props) * len(idx)).astype(int)[:-1]
            for part, cid in zip(np.split(idx, cuts), cids):
                out[cid].extend(part.tolist())
    return out


def partition_dataset(dataset, num_clients, alpha, mode="dirichlet"):
    if mode == "dirichlet":
        return dirichlet_partition(dataset, num_clients, alpha)
    if mode == "domain":
        return domain_partition(dataset, num_clients)
    if mode == "domain_dirichlet":
        return domain_dirichlet_partition(dataset, num_clients, alpha)
    raise ValueError(f"Unknown partition mode: {mode}")