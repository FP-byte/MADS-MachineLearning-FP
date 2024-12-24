1.1 Wat *tensors* zijn en voorbeelden voor de dimensies tm 5D
Tensor: vector of (2D) matrix van vectoren, (3D) (batch, hoogte, breedte), 4D (batch, hoogte, breedt, kanalen), 5D (batch, frames, hoogte, breedte, kanalen) bijv. van videostreaming
1.2  hoe neurale netwerken zijn opgebouwd (linear + activation), en hoe ze een extensie zijn van lineaire modellen (wanneer kun je ze stapelen)  
de functie y = f[x, params] is een opsommig van een line van de type Wx+b
1.3 neurale netwerken zijn een universal function
Een universele functiebenadering (universal function approximation) is een concept in de wiskunde en machine learning dat verwijst naar het vermogen van een model om elke continue functie te benaderen, mits het model voldoende complexiteit heeft.
Het idee is dat een model (zoals een neuraal netwerk) in staat is om elke willekeurige functie te benaderen tot een gewenste nauwkeurigheid, gegeven voldoende parameters, lagen, of tijd om te trainen.
Universal Approximation Theorem: een feedforward neuraal netwerk  met minstens één hidden layer (met voldoende aantal neuronen) kan een continue functie kan benaderen, op elke gesloten en begrensde interval, tot elke gewenste nauwkeurigheid. 
1.4 gradient descent: een optimalisatie-algoritme gebruikt in machine learning en deep learning om de parameters van een model (zoals de gewichten in een neuraal netwerk) aan te passen en zo de fout (of verlies) te minimaliseren. 

1.5 voordelen van een datastreamer voor grote datasets: data worden in batch aangeboden, om data in real-time of in kleine stukken (data chunks) te verwerken, in plaats van het laden van de volledige dataset in het geheugen.  Het biedt voordelen zoals efficiënt geheugengebruik, schaalbaarheid, real-time verwerking, en kostenbesparing, terwijl het ook in staat is om te werken met gedistribueerde gegevens en lage latentie biedt voor directe besluitvorming.
1.6 Wat de **invloed is van het aantal units in een dense layer** 
Meer units in een dense layer geven het model meer capaciteit om complexe relaties in de gegevens te leren. Elke unit in een laag kan als een soort "kenmerkdetector" worden gezien, dus meer units maken het model krachtiger in het herkennen van complexere patronen.
Als je te weinig units gebruikt, kan het model onderfitten (underfitting), wat betekent dat het model niet in staat is om de onderliggende patronen in de gegevens goed te leren.
Als je te veel units gebruikt, kan het model overfitten (overfitting), wat betekent dat het model te veel leert van de specifieke details in de trainingsgegevens en slecht generaliseert naar nieuwe, ongeziene gegevens.
Network capacity: aantal hidden units, meer hidden units meer benadering van complexe functies (width)
Network depth: aantal layers

Meer units leiden tot een groter aantal parameters in het netwerk, wat betekent dat er meer rekenkracht en geheugen nodig is voor zowel de training als de inferentie (voorspelling).
Dit verhoogt de trainingstijd, omdat het netwerk meer berekeningen moet uitvoeren tijdens elke stap van de gradient descent-optimalisatie.
De benodigde tijd voor backpropagation en gewicht-updates neemt toe naarmate het aantal units in een laag toeneemt.


1.7 **Hoe een gin file werkt** en wat de voordelen daarvan zijn: in een gin file worden de hyperparameters van een neurale netwerk mee gegeven bij het aanmaken van een netwerk. Daar kunnen de kenmerk ook aangepast worden voor hypertuning. 
1.8 Waarom we **train-test-valid splits** maken
Trainingsset: Wordt gebruikt om het model te trainen.
Validatieset: Wordt gebruikt om het model te evalueren en hyperparameters af te stemmen.
Testset: Wordt pas gebruikt na het trainen en afstemmen van het model om de uiteindelijke prestaties te testen.
Een gebruikelijke verdeling is bijvoorbeeld:
70% voor training
15% voor validatie
15% voor testen
Hyperparameters zijn instellingen van het model die niet tijdens het trainen worden geleerd, zoals de leersnelheid, het aantal lagen in een neuraal netwerk, het aantal eenheden in elke laag, de batchgrootte, en meer.
Je kunt de validatieset gebruiken om verschillende instellingen van hyperparameters uit te proberen en te testen om te zien welke het beste presteren. Dit proces wordt vaak hyperparameteroptimalisatie genoemd.
Een veelgebruikte aanpak om hyperparameters af te stemmen is grid search of random search, waarbij je verschillende combinaties van hyperparameters test en de configuratie kiest die de beste resultaten oplevert op de validatieset.
**Validatie**: Nadat het model is getraind, wordt de nauwkeurigheid geëvalueerd op de validatieset.
Dit helpt ons te begrijpen hoe goed het model presteert met de gegeven hyperparameters.
**Hyperparameteroptimalisatie**: Bijv. we testen verschillende leersnelheden (learning_rate_init) om te zien welke het beste werkt op de validatieset. Het model met de hoogste nauwkeurigheid op de validatieset wordt beschouwd als het beste model.
**Testen**: Het beste model wordt uiteindelijk geëvalueerd op de testset. De testset wordt pas gebruikt nadat de hyperparameters zijn geoptimaliseerd en het model is getraind op de trainingsset.

1.9 Wat een Activation function is, en kent er een aantal (ReLU, Leaky ReLU, ELU, GELU, Sigmoid) 
Een **activation function** (activeringsfunctie) is een wiskundige functie die de output van een neuron in een neuraal netwerk bepaalt. Het zorgt ervoor dat het netwerk niet-lineair (non-linear) wordt, waardoor het in staat is om complexe patronen en relaties in de data te leren.
*ReLU* is de meest gebruikte activatiefunctie in diepe netwerken. Het geeft de input door als de waarde groter is dan 0, en anders geeft het 0 terug. Het is eenvoudig en leidt tot snellere training, maar kan het probleem van "dode neuronen" veroorzaken, waarbij neuronen nooit geactiveerd worden als ze negatieve inputs ontvangen. 
*Leaky ReLU* is een variant van ReLU die voorkomt dat neuronen "doodgaan" door negatieve waarden een kleine, niet-nul output te geven (in plaats van 0). Dit zorgt ervoor dat de neuron ook voor negatieve inputs actief blijft.
*ELU* is een activatiefunctie die voor positieve waarden werkt zoals ReLU, maar voor negatieve waarden wordt de output een exponentiële functie die de negatieve output verzacht. Het helpt bij het verminderen van de bias van de activatiefunctie en het versnellen van de convergentie in vergelijking met ReLU.
*GELU* is een activatiefunctie die een combinatie is van een sigmoïde en een ReLU, en het heeft een gladder verloop dan ReLU. Het is gebaseerd op een probabilistische benadering en wordt vaak gebruikt in Transformer-netwerken zoals BERT. GELU is wiskundig geavanceerder dan ReLU, maar werkt vaak beter in moderne netwerken.
De *Sigmoidfunctie* is een S-vormige (sigmoïde) curve die de output beperkt tot het interval tussen 0 en 1. Het wordt vaak gebruikt in de outputlaag van classificatiemodellen voor binaire classificatie, omdat de output kan worden geïnterpreteerd als de waarschijnlijkheid van een klasse.
ReLU: Eenvoudig, snel, maar kan leiden tot "dode" neuronen.
Leaky ReLU: Voorkomt dode neuronen door een kleine negatieve helling te geven.
ELU: Een exponentiële variant die negatieve inputs verzacht, waardoor sneller convergeren mogelijk is.
GELU: Een gladde, probabilistische activatiefunctie die goed werkt in moderne netwerken zoals Transformers.
Sigmoid: Beperkt de output tussen 0 en 1, vaak gebruikt voor binaire classificatie.
1.10 wat een loss functie is:
Een loss functie (of kostenfunctie) berekent de fout of het verschil meet tussen de voorspelde waarden van een model en de werkelijke waarden (targets) in een dataset
Mean Squared Error (MSE):
Toepassing: Veel gebruikt voor regressieproblemen.
MSE berekent het gemiddelde van de kwadraten van de verschillen tussen de werkelijke en voorspelde waarden. Dit betekent dat grotere fouten zwaarder worden gestraft, wat kan helpen om grote afwijkingen te vermijden.
Cross-Entropy Loss (ook bekend als Log Loss): 
Deze loss functie meet de afwijking tussen de werkelijke klasse en de voorspelde waarschijnlijkheid. 
Toepassing: Veel gebruikt voor classificatieproblemen, vooral binaire en multi-class classificatie.
Afhankelijk van het type probleem (bijvoorbeeld classificatie of regressie) kies je een geschikte loss functie zoals Mean Squared Error (MSE) voor regressie of Cross-Entropy Loss voor classificatie.
1.11 Hoe deep learning past binnen de geschiedenis van AI, en wat deep learning kenmerkt ten opzichte van de rest.
First neuron idea is from 1943, perceptron (1958), 1980's backpropagatio developed which started a new development  
1.12 *stappen van het trainen van een NN*: datapreparatie, trainbare gewichten, predict, lossfunctie, optimizers 
A supervised learning model is a fuction y = f[x, params] that relates input x to input y by parameters. To train the model we define a loss function over a training dataset which quantifies the mismatch between the prediction of f and the observed outputs y. We search the parameters that minimize the loss. We evaluate on a different set of data (test) to see how well it generalizes. 