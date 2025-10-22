import os
import pickle
import json
import pandas as pd
import torch
import torch.nn as nn
from models.explainer import Explainer
from utils.data_handler import DataHandler
from utils.parse import args

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using device {device}")

class XRec:
    def __init__(self):
        print(f"Dataset: {args.dataset}")
        print(f"Dataset dir: {args.dataset_dir}")
        print(f"Output dir: {args.output_dir}")
        print(f"NL profiles dir: {args.profiles_dir}")

        self.model = Explainer().to(device)
        self.data_handler = DataHandler(
            domain=args.dataset,
            dataset_dir=args.dataset_dir,
            output_dir=args.output_dir,
            profiles_dir=args.profiles_dir,
        )

        self.trn_loader, self.val_loader, self.tst_loader = self.data_handler.load_data()
        self.user_embedding_converter_path = os.path.join(args.output_dir, "user_converter.pkl")
        self.item_embedding_converter_path = os.path.join(args.output_dir, "item_converter.pkl")

        self.output_path = os.path.join(args.output_dir, "output.csv")

    def train(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)
        for epoch in range(args.epochs):
            total_loss = 0
            self.model.train()
            for i, batch in enumerate(self.trn_loader):
                user_embed, item_embed, input_text = batch
                user_embed = user_embed.to(device)
                item_embed = item_embed.to(device)

                input_ids, outputs, explain_pos_position = self.model.forward(user_embed, item_embed, input_text)
                input_ids = input_ids.to(device)
                explain_pos_position = explain_pos_position.to(device)
                optimizer.zero_grad()
                loss = self.model.loss(input_ids, outputs, explain_pos_position, device)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                if i % 100 == 0 and i != 0:
                    print(
                        f"Epoch [{epoch}/{args.epochs}], Step [{i}/{len(self.trn_loader)}], Loss: {loss.item()}"
                    )
                    print(f"Generated Explanation: {outputs[0]}")

                if epoch == 0 and i == 0:
                    print(input_text[:min(5, len(input_text))])

            print(f"Epoch [{epoch}/{args.epochs}], Loss: {total_loss}")
            # Save the model
            torch.save(
                self.model.user_embedding_converter.state_dict(),
                self.user_embedding_converter_path,
            )
            torch.save(
                self.model.item_embedding_converter.state_dict(),
                self.item_embedding_converter_path,
            )
            print(f"Saved model to {self.user_embedding_converter_path}")
            print(f"Saved model to {self.item_embedding_converter_path}")

    def evaluate(self):
        loader = self.tst_loader

        # load model
        self.model.user_embedding_converter.load_state_dict(
            torch.load(self.user_embedding_converter_path)
        )
        self.model.item_embedding_converter.load_state_dict(
            torch.load(self.item_embedding_converter_path)
        )
        self.model.eval()
        predictions = []
        references = []
        with torch.no_grad():
            for i, batch in enumerate(loader):
                user_embed, item_embed, input_text, explain = batch
                user_embed = user_embed.to(device)
                item_embed = item_embed.to(device)

                outputs = self.model.generate(user_embed, item_embed, input_text)
                # LAST
                #end_idx = outputs[0].find("[")
                #if end_idx != -1:
                #    outputs[0] = outputs[0][:end_idx]
                #predictions.append(outputs[0])
                #references.append(explain[0])
                #

                # NEW
                for j in range(len(outputs)):
                    end_idx = outputs[j].find("[")
                    if end_idx != -1:
                        outputs[j] = outputs[j][:end_idx]

                    predictions.append(outputs[j])
                    references.append(explain[j])

                output_df = pd.DataFrame(
                    {
                        "prediction": predictions,
                        "reference": references,
                    }
                )

                output_df.to_csv(self.output_path, index=False)
                print(f"Saved predictions to {self.output_path}")

                if i % 10 == 0 and i != 0:
                    print(f"Step [{i}/{len(loader)}]")
                    print(output_df.tail(10))
        
        print("Evaluation completed.")


def main():
    sample = XRec()
    if args.mode == "finetune":
        print("Finetune model...")
        sample.train()
    elif args.mode == "generate":
        print("Generating explanations...")
        sample.evaluate()

if __name__ == "__main__":
    main()
