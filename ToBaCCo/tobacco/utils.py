import numpy as np
import re


PT = ['H' , 'He', 'Li', 'Be', 'B' , 'C' , 'N' , 'O' , 'F' , 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P' , 'S' , 'Cl', 'Ar',
	  'K' , 'Ca', 'Sc', 'Ti', 'V' , 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
	  'Rb', 'Sr', 'Y' , 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I' , 'Xe',
	  'Cs', 'Ba', 'Hf', 'Ta', 'W' , 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 
	  'Ra', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Ac', 'Th', 
	  'Pa', 'U' , 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'FG', 'X' ]

vname_dict = {'V':1,'Er':2,'Ti':3,'Ce':4,'S':5,
			  'H':6,'He':7,'Li':8,'Be':9,'B':10,
			  'C':11,'N':12,'O':13,'F':14,'Ne':15,
			  'Na':16,'Mg':17,'Al':18,'Si':19,'P':20 ,
			  'Cl':21,'Ar':22,'K':23,'Ca':24,'Sc':24,
			  'Cr':26,'Mn':27,'Fe':28,'Co':29,'Ni':30}

metal_elements = ['Ac','Ag','Al','Am','Au','Ba','Be','Bi',
				  'Bk','Ca','Cd','Ce','Cf','Cm','Co','Cr',
				  'Cs','Cu','Dy','Er','Es','Eu','Fe','Fm',
				  'Ga','Gd','Hf','Hg','Ho','In','Ir',
				  'K','La','Li','Lr','Lu','Md','Mg','Mn',
				  'Mo','Na','Nb','Nd','Ni','No','Np','Os',
				  'Pa','Pb','Pd','Pm','Pr','Pt','Pu','Ra',
				  'Rb','Re','Rh','Ru','Sc','Sm','Sn','Sr',
				  'Ta','Tb','Tc','Th','Ti','Tl','Tm','U',
				  'V','W','Y','Yb','Zn','Zr']

def PBC3DF(c1, c2):

    diffa = c1[0] - c2[0]
    diffb = c1[1] - c2[1]
    diffc = c1[2] - c2[2]

    if diffa > 0.5:
        c2[0] = c2[0] + 1.0
    elif diffa < -0.5:
        c2[0] = c2[0] - 1.0
    
    if diffb > 0.5:
        c2[1] = c2[1] + 1.0
    elif diffb < -0.5:
        c2[1] = c2[1] - 1.0
 
    if diffc > 0.5:
        c2[2] = c2[2] + 1.0
    elif diffc < -0.5:
        c2[2] = c2[2] - 1.0

    return c2

def PBC3DF_sym(vec1, vec2, usage):

    dX,dY,dZ = vec1 - vec2
            
    if dX > 0.5:
        s1 = 1.0
        ndX = dX - 1.0
    elif dX < -0.5:
        s1 = -1.0
        ndX = dX + 1.0
    else:
        s1 = 0.0
        ndX = dX
                
    if dY > 0.5:
        s2 = 1.0
        ndY = dY - 1.0
    elif dY < -0.5:
        s2 = -1.0
        ndY = dY + 1.0
    else:
        s2 = 0.0
        ndY = dY

    if dZ > 0.5:
        s3 = 1.0
        ndZ = dZ - 1.0
    elif dZ < -0.5:
        s3 = -1.0
        ndZ = dZ + 1.0
    else:
        s3 = 0.0
        ndZ = dZ

    if usage == "adjust_edges":
        sym = np.array([s1,s2,s3])
        return np.array([ndX,ndY,ndZ]), sym
    elif usage == "write_cifs":
        if str(s1) + str(s2) + str(s3) == '555':
            sym = '.'
        else:
            sym = '1_' + str(s1) + str(s2) + str(s3)
        return np.array([ndX,ndY,ndZ]), sym

def nn(string):
	return re.sub('[^a-zA-Z]','', string)

def nl(string):
	return re.sub('[^0-9]','', string)

