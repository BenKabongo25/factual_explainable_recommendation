# Ben Kabongo
# December 2024

from torch.utils.data import Dataset
from tqdm import tqdm


class RatingReviewDataset(Dataset):
    
    def __init__(self, config, data_df, tokenizer):
        super().__init__()
        self.data_df = data_df
        self.tokenizer = tokenizer
        self.config = config
        self.reviews = self.data_df["review"].tolist()
        self.reviews_tokens = []
        self.reviews_masks = []
        self._process()
    
    def __len__(self) -> int:
        return len(self.data_df)
        
    def _process(self):
        for index in tqdm(range(len(self)), desc="Processing data", colour="green"):
            review = str(self.reviews[index])
            self.reviews[index] = review

            inputs = self.tokenizer(
                f"{self.config.bos} {review} {self.config.eos}",
                max_length=self.config.review_length,
                truncation=True, 
                padding="max_length",
                return_tensors="pt"
            )
            tokens = inputs["input_ids"].squeeze(0)
            #tokens[tokens == self.tokenizer.pad_token_id] = -100
            masks = inputs["attention_mask"].squeeze(0)
            self.reviews_tokens.append(tokens)
            self.reviews_masks.append(masks)
                
    def __getitem__(self, index):
        row = self.data_df.iloc[index]

        user_id = row["user_id"]
        item_id = row["item_id"]
        rating = row["rating"]

        review = self.reviews[index]
        tokens = self.reviews_tokens[index]
        mask = self.reviews_masks[index]
                        
        return {
            "user_id": user_id,
            "item_id": item_id,
            "rating": rating,
            "review": review,
            "tokens": tokens,
            "mask": mask,
        }