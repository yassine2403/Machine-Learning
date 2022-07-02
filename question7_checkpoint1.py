sqrt= lambda x: x**0.5
def func(D):
    C=50
    H=30
    f=lambda d: round(sqrt((2*C*d)/H))
    outputlist=list(map(f,D))
    return outputlist
print(func([100,150,180]))
