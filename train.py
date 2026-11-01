"""Transfer learning klasifikasi: bandingkan feature | partial | scratch.
Contoh : python train.py --mode all --arch resnet18 --epochs 10
Hasil  : results/log_*.csv, results/summary.csv, results/curve.png, best_<arch>_<mode>.pt"""
import argparse

import torch

from vision import cls_data, cls_models, cls_train


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data", default="dataset_raw"); ap.add_argument("--split", default="split.json")
    ap.add_argument("--arch", default="resnet18", choices=cls_models.ARCHS)
    ap.add_argument("--mode", default="all", choices=list(cls_models.MODES) + ["all"])
    ap.add_argument("--epochs", type=int, default=10); ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    sp = cls_data.load_split(a.split)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", dev, "| kelas:", sp["classes"], f"| train={len(sp['train'])} val={len(sp['val'])}")
    tr_dl, va_dl = cls_data.make_loaders(a.data, sp, a.batch)
    modes = list(cls_models.MODES) if a.mode == "all" else [a.mode]
    results = [cls_train.fit_mode(a.arch, m, sp["classes"], tr_dl, va_dl, a.epochs, dev, a.seed) for m in modes]
    cls_train.write_logs(results)
    print("\narch\tmode\tbest_val\tep_best\tep>=90%\tdetik")
    for r in results:
        print(f"{r.arch}\t{r.mode}\t{r.best_val:.4f}\t{r.best_epoch}\t{r.first_90 or '-'}\t{r.seconds:.0f}")
    print("Grafik: results/curve.png" if cls_train.plot_curves(results) else "matplotlib belum terpasang; grafik dilewati")


if __name__ == "__main__":
    main()
