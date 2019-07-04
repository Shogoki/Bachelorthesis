<!--# Abbildungsverzeichnis {.unnumbered}-->

<!--
For me, this was the only drawback of writing in Markdown: it is not possible to add a short caption to figures and tables. This means that the \listoftables and \listoffigures commands will generate lists using the full titles, which is probably isn't what you want. For now, the solution is to create the lists manually, when everything else is finished.
-->

<!--Abbildung 4.1  Ein Beispieö . . .              \hfill{pp}  
<!-- Figure x.x  Short title of the figure . . .         \setcounter{page}{3}     \hfill{pp}   -->
\clearpage
\pagenumbering{roman}

# Abbildungsverzeichnis {.unnumbered}

\renewcommand{\chaptermark}[1]{\markboth{}{\sffamily #1}}
\chaptermark{Abbildungsverzeichnis}

Abbildung \ref{ki_ml_dl} Einordnung in die künstliche Intelligenz . . . \hfill{\pageref{ki_ml_dl}}  \newline
Abbildung \ref{bio_neuron} Ein biologisches Neuron . . .  \setcounter{page}{3} \hfill{\pageref{bio_neuron}} \newline
Abbildung \ref{perzeptron} Perzeptron . . . \hfill{\pageref{perzeptron}} \newline
Abbildung \ref{aktfunktionen} Aktivierungsfunktionen . . . \hfill{\pageref{aktfunktionen}} \newline
Abbildung \ref{simple_nn} verinfachtes neuronales Netzwerk . . . \hfill{\pageref{simple_nn}} \newline
Abbildung \ref{hunde_klassifizierer} beispielhafter Hunde/Katzen-Klassifizierer . . . \hfill{\pageref{hunde_klassifizierer}} \newline
Abbildung \ref{CNN} neuronales Faltungsnetzwerk . . . \hfill{\pageref{CNN}} \newline
Abbildung \ref{haar_features} Haar-Merkmale . . . \hfill{\pageref{haar_features}} \newline
Abbildung \ref{integral_image} Berechnung eines Integralbildes . . . \hfill{\pageref{integral_image}} \newline
Abbildung \ref{integral_image_reuse} weiterführende Berechnung eines Integralbildes . . . \hfill{\pageref{integral_image_reuse}} \newline
Abbildung \ref{viola_jones_adaboost} modifizierter AdaBoost Algorithmus . . . \hfill{\pageref{viola_jones_adaboost}} \newline
Abbildung \ref{cascade_of_classifiers} kaskadierende Klassifizierer . . . \hfill{\pageref{cascade_of_classifiers}} \newline
Abbildung \ref{single_ferplus} Datum aus dem FER+ Datensatz . . . \hfill{\pageref{single_ferplus}} \newline
Abbildung \ref{webrtc_screenshot} Webservice zum aufzeichnen der Videos . . . \hfill{\pageref{webrtc_screenshot}} \newline
Abbildung \ref{data_split} Aufteilung der Datensätze . . . \hfill{\pageref{data_split}} \newline
Abbildung \ref{architecture_simple_cnn} Architektur des einfachen Faltungsnetzwerkes . . . \hfill{\pageref{architecture_simple_cnn}} \newline
Abbildung \ref{avg_pooling} Durchnitss-Pooling . . . \hfill{\pageref{avg_pooling}} \newline
Abbildung \ref{sepconv} separable convolution . . . \hfill{\pageref{sepconv}} \newline
Abbildung \ref{xception_architektur} Architektur des Xception Netzwerkes . . . \hfill{\pageref{xception_architektur}} \newline
Abbildung \ref{simple_cnn_training} Trainingsverlauf des einfachen Faltungsnetzwerkes . . . \hfill{\pageref{simple_cnn_training}} \newline
Abbildung \ref{best_xcepton_training} Trainingsverlauf des besten Models . . . \hfill{\pageref{best_xcepton_training}} \newline
Abbildung \ref{more_regularization_acc} Genauigkeit des Netzwerks mit mehr Regularisierung . . . \hfill{\pageref{more_regularization_acc}} \newline
Abbildung \ref{more_regularization_loss} Verlust des Netzwerks mit mehr Regularisierung . . . \hfill{\pageref{more_regularization_loss}} \newline
Abbildung \ref{app_architecture} Softwarearchitektur des erstellten Webservice . . . \hfill{\pageref{app_architecture}} \newline
Abbildung \ref{frontend} Abbildung der minimalsitischen Benutzerschnittstelle . . . \hfill{\pageref{frontend}} \newline
Abbildung \ref{conf_matrix_bridge} Verwirrungsmatrix für den Bridge Datensatz . . . \hfill{\pageref{conf_matrix_bridge}} \newline
Abbildung \ref{conf_matrix_test} Verwirrungsmatrix für den Test Datensatz . . . \hfill{\pageref{conf_matrix_test}} \newline
Abbildung \ref{screenshot_testweb} Test des Webservice . . . \hfill{\pageref{screenshot_testweb}} \newline

\newpage