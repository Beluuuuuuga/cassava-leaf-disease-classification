
def show_inheritance(cls):
    inheritance = str(cls.__mro__)
    print("タスクの継承関係：",inheritance.replace("<class \'__main__.", '\'').replace(">, <class \'object\'>)",'').replace(">,","<-")[1:])