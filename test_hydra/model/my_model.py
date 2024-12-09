class MyModel():
    def __init__(self, 
                 param1,
                 param2,
                 param3,
                 **kwargs
                 ):
        self.constant_1=param1
        self.constant_2=param2
        self.constant_3=param3
    
    def foo(self):
        print(self.constant_1)
        print(self.constant_2)
        print(self.constant_3)
