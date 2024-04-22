from happytransformer import TTTrainArgs
from happytransformer import HappyTextToText
import pickle

# load the model
happy_tt = HappyTextToText("T5", "csebuetnlp/banglat5")

before_result = happy_tt.eval("eval.csv")
print("Before loss:", before_result.loss)

args = TTTrainArgs(batch_size=1,num_train_epochs=6)
happy_tt.train("train.csv", args=args)


after_loss = happy_tt.eval("eval.csv")
print("After loss: ", after_loss.loss)

# save the trained model
pickle.dump(happy_tt, open('model.pkl','wb'))