from happytransformer import TTTrainArgs
from happytransformer import HappyTextToText
from happytransformer import TTSettings
import pickle

model = pickle.load(open('model.pkl', 'rb'))


beam_settings =  TTSettings(num_beams=5, min_length=1, max_length=32)

example_1 = "grammar: আজ রোফবার দুপুরে রাজফনীর ইস্কাটনে ঢাকা ম্যাস ট্রানজিট কোম্পাকি লিমিডোডের (ডিএমটিসিএল) কার্যালয়ে আয়োজিএ এক সংবাজ সম্মেলনে এ কতা জানাদ সংস্থাটির ব্যবস্থাপকা পরিচাকক"
result_1 = model.generate_text(example_1, args=beam_settings)
print(result_1.text)