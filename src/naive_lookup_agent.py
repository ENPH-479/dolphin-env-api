import pickle


class NaiveAgent:
    def __init__(self, pickled_model_path):
        saved_file = open(pickled_model_path, 'rb')
        saved_obj = pickle.load(saved_file)
        self.model = saved_obj.get("model")
        self.defaults = saved_obj.get("defaults")
        saved_file.close()

    def run(self):
        # TODO complete
        # take screenshot
        # load img
        # downsample
        # look up game state
        # generate random number [0,1]
        # take diff of key press state - update with current pressed keys
        # update controller inputs through pipe
        # delete screenshot?
        pass
