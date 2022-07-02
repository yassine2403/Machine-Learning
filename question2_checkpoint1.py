#regular factorial function
def factorial(n):
    facto=1
    if n==0:
        return(1)
    else:
        for i in range(1,n+1):
            facto*=i
    return facto
#recursive definition of factorial
def rfactorial(n):
    if n==0:
        return 1
    return n*factorial(n-1)

