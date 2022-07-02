def missing_char(char,n):
    emptychar=''
    for i in range(len(char)):
        if i!=n:
            emptychar+=char[i]
    return emptychar
