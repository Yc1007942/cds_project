export interface AnalysisGalleryItem {
  slug: string;
  title: string;
  image: string;
  description: string;
  finding: string;
}

export const analysisGallery: AnalysisGalleryItem[] = [
  {
    slug: "train-val-accuracy",
    title: "Training vs Validation Accuracy",
    image: "/analysis/train_val_acc.png",
    description:
      "Accuracy curves across training epochs, comparing the model's performance on the training split against the held-out validation split.",
    finding:
      "Accuracy rises quickly and then stabilizes, which suggests the classifier converges early and generalizes without a large train-validation gap.",
  },
  {
    slug: "train-val-f1",
    title: "Training vs Validation F1 Score",
    image: "/analysis/train_val_f1.png",
    description:
      "Macro F1 progression across epochs, used to verify class-sensitive performance instead of relying on accuracy alone.",
    finding:
      "Validation F1 tracks training F1 closely, indicating the model is learning a balanced decision boundary rather than over-optimizing one class.",
  },
  {
    slug: "train-val-loss",
    title: "Training vs Validation Loss",
    image: "/analysis/train_val_loss.png",
    description:
      "Loss curves for both training and validation runs, showing optimization stability as the model learns.",
    finding:
      "Loss falls and then flattens rather than diverging, which supports the claim that the training process remains stable through later epochs.",
  },
  {
    slug: "autoint-top-features",
    title: "AutoInt Top 20 Features",
    image: "/analysis/top_20_autoInt.png",
    description:
      "Top twenty ranked features from the AutoInt model, highlighting which handcrafted inputs contribute most strongly to classification.",
    finding:
      "The ranking surfaces a compact set of dominant linguistic signals, which is useful for explaining where the model is finding separation between AI and human text.",
  },
  {
    slug: "ft-transformer-top-features",
    title: "FT-Transformer Top 20 Features",
    image: "/analysis/ft_trans_top20_features.png",
    description:
      "Top twenty feature importances from the FT-Transformer model, showing which structured inputs remain most influential after transformer-based mixing.",
    finding:
      "The FT-Transformer emphasizes a slightly different feature mix, which helps compare whether importance patterns are model-specific or consistent across architectures.",
  },
];
