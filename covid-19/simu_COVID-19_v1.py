import os
import sys
import numpy as np
import matplotlib.pyplot as plt


def ode_system_print_dynamic(out_, t_, y_) :
    out_.write("{:16.14e} ".format(t_))
    for i in range(len(y_)) :
        out_.write("{:16.14e} ".format(y_[i]))
    out_.write("\n")
    return

def ode_rk4_stepper(t_, y_, dt_, func_derivatives_, params_) :
	# Variables de  travail:
	ndyn = len(y_)
	# Tableau temporaire des variables dynamiques courantes
	yt = y_[0:]

	### Etape 1:
	# Calcul des dérivées:
	dydt = func_derivatives_(t_, yt, params_)
	# Calcul des variations:
	ky1 = [0.0] * ndyn
	for i in range(ndyn) :
		ky1[i] = dt_ * dydt[i]

	### Etape 2:
	for i in range(ndyn) :
		yt[i] = y_[i] + 0.5 * ky1[i]
	dydt = func_derivatives_(t_ + 0.5 * dt_, yt, params_)
	ky2 = [0.0] * ndyn
	for i in range(ndyn) :
		ky2[i] = dt_ * dydt[i]

	### Etape 3:
	for i in range(ndyn) :
		yt[i] = y_[i] + 0.5 * ky2[i]
	dydt = func_derivatives_(t_ + 0.5 * dt_, yt, params_)
	ky3 = [0.0] * ndyn
	for i in range(ndyn) :
		ky3[i] = dt_ * dydt[i]

	### Etape 4:
	for i in range(ndyn) :
		yt[i] = y_[i] + ky3[i]
	dydt = func_derivatives_(t_ + dt_, yt, params_)
	ky4 = [0.0] * ndyn
	for i in range(ndyn) :
		ky4[i] = dt_ * dydt[i]

	# Calcul des nouvelles variables dynamiques:
	tf = t_ + dt_
	yf = y_[0:]
	for i in range(ndyn) :
		yf[i] += (ky1[i] + 2 * (ky2[i] + ky3[i]) + ky4[i]) / 6.0
	return (tf, yf)



# Bloc principal:
if __name__ == "__main__" :

	# Variables d'entrée du modèle RSI
	S = 0.9
	I = 0.1
	R = 0.
	M = 0.

	def fct_derivatives(t_, y_, osc_) :
		# Extraction des paramètres statiques de l'oscillateur:
		beta = osc_[0] # Facteur Beta
		lamb = osc_[1]
		mu   = osc_[2]
		# Extraction des variables dynamiques du système:
		s = y_[0] # Saine
		i = y_[1] # Infectée
		r = y_[2] # Rétablie
		m = y_[3]
		# Calcul des dérivées des variables dynamiques:
		dsdt = -beta*s*i                    # Dérivée de la pop saine
		didt = beta*s*i - (1/lamb)*i - mu*i # Dérivée de la pop infecté
		drdt = (1/lamb)*i	                # Dérivée de la pop rétablie
		dmdt = mu*i                         # Dérivée de la pop morte
		# Construction du tableau des dérivées instantanées des variables dynamiques:
		dydt = []
		dydt.append(dsdt)
		dydt.append(didt)
		dydt.append(drdt)
		dydt.append(dmdt)
		return dydt

	# Définition des paramètres du système :
	BETA   = 0.5
	LAMBDA = 2.0
	MU     = 0.02
	osc = (BETA,LAMBDA,MU)

	# Fichier de sauvegarde des données:
	data_out = open("RSI_model.data", "w")

	# Conditions initiales:
	t0 = 0.0  # Temps

	# Tableau des variables dynamiques du système:
	t = t0
	y = [S, I, R, M]
	ode_system_print_dynamic(data_out, t, y)
	tmax = 50.0
	#delta_t = tmax / int(input_nsteps)
	delta_t = 0.01
	input_nsteps = tmax/delta_t

	pop  = 1000
	step = 0
	derivatives_values = np.zeros((int(input_nsteps)+2, len(y)+1))
	derivatives_values[step][0] = t
	for h in range(len(y)):
		derivatives_values[step][h+1] = int(y[h]*pop)

	while tmax > t :
		step += 1
		(t, y) = ode_rk4_stepper(t, y, delta_t, fct_derivatives, osc)
		derivatives_values[step][0] = t
		for j in range(len(y)):
			derivatives_values[step][j+1] = int(y[j]*pop)
		ode_system_print_dynamic(data_out, t, y)
	
	# Vérification que personne ne s'est perdu en chemin
	nb_end = np.sum(derivatives_values[int(input_nsteps)])-tmax
	print("-> debug :: Il y a {:5.5e} à la fin de la simulation."
		.format(nb_end))
	if nb_end != pop:
		print("-> warning :: Il manque {:3.0e} personnes !".format(np.abs(nb_end - pop)))
	else :
		print("-> debug :: Tout est bon, circulez, il n'y a rien à voir.")

	data_out.close()

	# Affichage des graphes d'évolution des eq. différentiels
	plt.plot(derivatives_values[:,1], label="Personnes saines")
	plt.plot(derivatives_values[:,2], label="Personnes malade")
	plt.plot(derivatives_values[:,3], label="Personnes guéries")
	plt.plot(derivatives_values[:,4], label="Personnes décédées")
	plt.legend()
	plt.show()


	sys.exit(0)

