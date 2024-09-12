from dataset import TwitterDataModule
import config
from transformers import AutoTokenizer
import pytorch_lightning as pl
from model import EmoTweetClassifier

if __name__ == '__main__':
    dm =  TwitterDataModule(data_dir=config.dir_data, 
                            tokenizer=config.tokenizer, 
                            max_len_token=config.max_len_token,
                            batch_size=config.batch_size,
                            num_workers=config.num_workers, 
                            train_split=config.train_split,
                            val_split=config.val_split,
                            test_split=config.test_split)



    model = EmoTweetClassifier(input_size=config.input_size, learning_rate = config.learning_rate, num_classes=config.num_classes)
    
    trainer = pl.Trainer(accelerator= config.accelerator, devices = config.devices, min_epochs=1, max_epochs=config.num_epochs, precision=config.percision)

    trainer.fit(model, dm)
    trainer.validate(model, dm)
    trainer.test(model, dm)
   
    print("done")