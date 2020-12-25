class Scaffold():
    def __init__(self, debug=True):
        self.on = debug

    def debug(self):
        self.on = True
    
    def prod(self):
        self.on = False

    def print(self, *msg):
        if self.on:
            print(*msg)
