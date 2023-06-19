import numpy as np

def conditionDict(inputDict, arrayOfKeys, arrayOfNorms = 1):
	'''
	Riceve in ingresso un dizionario inputDict e restituisce in uscita una lista
	contenente gli elementi di inputDict corrispondenti alle chiavi indicate nell'arrayOfKeys
	e normate rispetto a arrayOfNorms
	'''
	new_dict = {key:inputDict[key] for key in arrayOfKeys}
	
	var = []
	for k,v in new_dict.items():
		if type(v) is type([]):
			var.extend(v)
		else:
			var.append(v)

	var = np.array(var, dtype='float32')
	arrayOfNorms = np.array(arrayOfNorms, dtype='float32')
	
	return var/arrayOfNorms