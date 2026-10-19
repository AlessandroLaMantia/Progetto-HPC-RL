from pathlib import Path
import html
import re
import json

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'relazione'
SOURCE = Path(r'C:\Users\LENOVO\.codex\attachments\5af8444d-d4cd-4e87-b90e-5f6242250fd7\Testo incollato.txt')
slides = re.findall(r'<section\b.*?</section>', SOURCE.read_text(encoding='utf-8'), re.S)
parts = []
FUNCTION_FILES = {
    'load_target_state': 'quantum_rl/1.targets.py',
    'validate_target_state': 'quantum_rl/1.targets.py',
    'reset': 'quantum_rl/2.environment.py',
    'step': 'quantum_rl/2.environment.py',
    '_build_action_catalogue': 'quantum_rl/2.environment.py',
    '_state_torch': 'quantum_rl/2.environment.py',
    '_simulate': 'quantum_rl/2.environment.py',
    '_distance': 'quantum_rl/2.environment.py',
    '_get_obs': 'quantum_rl/2.environment.py',
    '_action_mask': 'quantum_rl/2.environment.py',
    '_optimise_angles': 'quantum_rl/2.environment.py',
    '_get_info': 'quantum_rl/2.environment.py',
    'forward': 'quantum_rl/3.model.py',
    'value_function': 'quantum_rl/3.model.py',
    'save_complete_checkpoint': '2.train.py',
    'evaluate_best_circuit': '2.train.py',
    'run_training_once': '3.run_repeated_training.py',
    'validate_checkpoint': '5.evaluate.py',
    'run_random_circuit': '1.pre_trained_table.py',
    'load_summary': '4.plot_training_summary.py',
    'load_training_history': '4.plot_training_summary.py',
}
def p(text):
    for name, source in FUNCTION_FILES.items():
        text = re.sub(r'(?<![\w.])'+re.escape(name)+r'\([^)]*\)', lambda m: '<code>'+m.group(0)+'</code> ('+source+')', text)
    parts.append('<p>'+text+'</p>')
def h(text):
    chapter = ' class="chapter"' if re.match(r'^\d+\.', text) else ''
    parts.append('<h2'+chapter+'>'+text+'</h2>')
def sub(text): parts.append('<h3>'+text+'</h3>')
def code(text, source=None):
    if source:
        parts.append('<p class="code-source">'+source+'</p>')
    parts.append('<pre>'+html.escape(text.strip())+'</pre>')
def table(headers, rows):
    parts.append('<table><thead><tr>'+''.join('<th>'+str(c)+'</th>' for c in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+str(c)+'</td>' for c in row)+'</tr>' for row in rows)+'</tbody></table>')
def clean(text): return html.unescape(re.sub('<[^>]+>', '', text)).strip()

parts.append('''<div class="cover"><div class="university">Università di Parma<br>Corso di High Performance Computing</div><h1>Sintesi di circuiti quantistici<br>con PPO, Transformer e HPC</h1><div class="subtitle">Relazione sul progetto</div><div class="author">Alessandro La Mantia<br><span class="student-id">Matricola: 399269</span></div></div>''')
h('Introduzione')
p('In questo progetto ho studiato come costruire un circuito quantistico che prepari uno stato desiderato usando pochi gate. Il circuito parte sempre da |00…0⟩. Ho usato un agente di reinforcement learning per scegliere i gate e un ottimizzatore per regolare gli angoli delle rotazioni.')
p('Ho lavorato su Bell, GHZ e due stati target random, a due e tre qubit. Per ogni caso ho confrontato i risultati con circuiti casuali. Ho usato il cluster HPC per eseguire gli addestramenti. In questa relazione spiego la struttura del progetto, il funzionamento del codice e i risultati che ho riportato nella presentazione.')
parts.append('<nav><h2>Indice</h2><ol>'+''.join('<li>'+x+'</li>' for x in ['Obiettivo e strumenti','Struttura delle cartelle','Caricamento dei target','Ambiente e azioni','Calcolo della reward','Ottimizzazione degli angoli','Transformer e PPO','Training e valutazione','Esecuzione sul cluster HPC','Risultati','Conclusioni e possibili miglioramenti'])+'</ol></nav>')
h('1. Obiettivo e strumenti')
p('Lo stato target è lo stato quantistico che voglio ottenere. Per misurare quanto il circuito si avvicina al target uso la fidelity F, che varia tra 0 e 1. Un valore vicino a 1 indica un buon risultato. Nel codice uso anche l’infedeltà d = 1 − F: più è bassa, meglio è.')
code('F = |⟨ψ_target | ψ_circuito⟩|²\nd = 1 − F\nSuccesso nell’ambiente: d < 0,01')
p('Nei report considero riusciti i tentativi con F ≥ 0,99. L’ambiente termina quando d &lt; 0,01, quindi con F &gt; 0,99. L’obiettivo è raggiungere questa precisione con pochi gate. La reward premia entrambi gli aspetti, ma non garantisce di trovare sempre il circuito più corto possibile.')
table(['Strumento','Uso nel progetto'],[['Gymnasium','Definisco ambiente, azioni e osservazioni.'],['PennyLane','Simulo i circuiti e calcolo i gradienti degli angoli.'],['PyTorch','Implemento la rete e ottimizzo gli angoli con Adam.'],['Ray RLlib','Uso PPO e raccolgo esperienze con più worker.'],['Qiskit','Genero la baseline di circuiti casuali.'],['SLURM','Invio gli esperimenti al cluster HPC.']])
p('Ho usato simulazioni classiche di stati puri, senza rumore. Non ho eseguito i circuiti su un computer quantistico reale. Circuiti brevi restano comunque interessanti perché, su un dispositivo reale, ogni gate può introdurre errori.')
h('2. Struttura delle cartelle')
p('Ho diviso il progetto in cinque script principali e tre moduli nella cartella quantum_rl. Gli script gestiscono gli esperimenti; i moduli contengono target, ambiente e rete.')
code('''Definitivo/
├── 1.pre_trained_table.py
├── 2.train.py
├── 3.run_repeated_training.py
├── 4.plot_training_summary.py
├── 5.evaluate.py
├── quantum_rl/
│   ├── __init__.py
│   ├── 1.targets.py
│   ├── 2.environment.py
│   └── 3.model.py
├── targets/
│   ├── bell.json
│   ├── ghz.json
│   ├── random_target_2_qubit_normalized.json
│   ├── random_target_3_qubit_normalized.json
│   ├── random_target_not_normalized.json
│   └── invalid_wrong_dimension_target.json
├── plots/
│   ├── bell/
│   ├── ghz/
│   ├── random2/
│   ├── random3/
│   └── matplotlib_cache/
├── checkpoints/
├── environment-hpc.yml
├── requirements.txt
├── job_train.slurm
├── README.md
├── .vscode/
├── __pycache__/
├── pdf_browser_audit/
└── relazione/''')
table(['File','A cosa serve'],[['1.pre_trained_table.py','Costruisce circuiti casuali e salva fidelity, gate, esito e tempo in un CSV.'],['2.train.py','Avvia PPO, registra le metriche e salva i checkpoint.'],['3.run_repeated_training.py','Ripete training e valutazione più volte e crea una tabella finale.'],['4.plot_training_summary.py','Legge i CSV e crea i grafici di reward e fidelity.'],['5.evaluate.py','Carica una policy addestrata e misura i risultati su più episodi.'],['quantum_rl/1.targets.py','Carica, controlla e normalizza lo stato target.'],['quantum_rl/2.environment.py','Gestisce circuito, azioni, angoli, simulazione e reward.'],['quantum_rl/3.model.py','Definisce il Transformer che propone le azioni.'],['quantum_rl/__init__.py','Permette di importare i moduli numerati con nomi semplici.']])
p('In targets salvo gli stati da preparare. In plots raccolgo i grafici, separati per esperimento. I checkpoint servono a salvare e ricaricare l’agente. requirements.txt elenca le librerie; environment-hpc.yml definisce l’ambiente Conda; job_train.slurm contiene il comando da eseguire sul cluster.')
p('Le cartelle .vscode, __pycache__, matplotlib_cache e pdf_browser_audit contengono impostazioni o file di supporto. Non fanno parte dell’algoritmo. In relazione salvo questo documento e i file usati per generarlo.')
p('I nomi dei moduli iniziano con un numero. Per importarli, __init__.py usa importlib e li espone come quantum_rl.targets, quantum_rl.environment e quantum_rl.model. In questo modo mantengo un solo file sorgente per ciascun modulo.')
h('3. Caricamento dei target')
p('Ho previsto tre tipi di target. Bell usa due qubit; GHZ può usarne più di due; un target custom viene letto da file. Nei miei esperimenti ho usato GHZ a tre qubit e due file random fissati, uno a due e uno a tre qubit.')
code('Bell = (|00⟩ + |11⟩) / √2\nGHZ₃ = (|000⟩ + |111⟩) / √2')
p('La funzione load_target_state() legge file JSON o NumPy (.npy). Nel JSON ogni ampiezza può essere un numero reale oppure una coppia [parte reale, parte immaginaria]. Questo esempio rappresenta Bell:')
code('[[0.7071067811865475, 0], [0, 0],\n [0, 0], [0.7071067811865475, 0]]')
p('validate_target_state() converte i dati in un vettore complesso e controlla la dimensione. Per n qubit servono 2ⁿ ampiezze: quattro per due qubit, otto per tre. Controlla anche che la norma sia finita e diversa da zero, poi divide il vettore per la norma. Posso quindi fornire anche un vettore non normalizzato.')
code('vector = np.asarray(state, dtype=np.complex128).reshape(-1)\nnorm = np.linalg.norm(vector)\n# Dopo i controlli su dimensione e norma:\nreturn vector / norm, inferred_qubits', 'quantum_rl/1.targets.py — estratto di validate_target_state(); np.asarray e np.linalg.norm sono funzioni NumPy.')
p('Se specifico sia il file sia il nome di un target, il programma usa il file. Il numero di qubit viene dedotto dalla dimensione e confrontato con quello eventualmente indicato da riga di comando.')
h('4. Ambiente e azioni')
sub('4.1 Come rappresento il circuito')
p('La classe QuantumCircuitEnv, in quantum_rl/2.environment.py, contiene il circuito corrente. In current_gates salvo la sequenza dei gate; in rotation_angles salvo gli angoli di RX, RY e RZ. Ogni gate è descritto da GateAction, definita nello stesso file, con nome e qubit coinvolti.')
table(['Gate','Scelta dell’agente','Angolo'],[['RX, RY, RZ','Tipo di rotazione e qubit','Lo ottimizzo con Adam.'],['H','Qubit su cui applicare Hadamard','Non ha un parametro da ottimizzare.'],['CNOT','Qubit di controllo e qubit target','Non ha un parametro da ottimizzare.']])
p('Ogni azione è un numero intero che identifica un gate del catalogo. Per n qubit ci sono 3n rotazioni, n Hadamard e n(n−1) CNOT diretti: dieci azioni a due qubit e diciotto a tre. L’agente sceglie il gate, mentre gli angoli vengono calcolati separatamente.')
sub('4.2 Un episodio, dall’inizio alla fine')
p('Per spiegare il ciclo considero un esempio: voglio preparare Bell. Ho quindi un target fisso, (|00⟩ + |11⟩)/√2, ma il circuito parte da |00⟩. Il target è il riferimento con cui confronto i tentativi: non lo applico direttamente ai qubit.')
sub('Primo passaggio: preparo il punto di partenza')
p('Chiamo reset(). Cancello i gate del tentativo precedente, svuoto gli angoli e simulo il circuito vuoto. Ottengo lo stato |00⟩. Chiamo poi _distance() per misurare l’errore iniziale: per Bell la fidelity è 0,5 e l’infedeltà è 0,5. Salvo questo errore in prev_dist, perché mi servirà dopo il primo gate.')
code('self.current_gates = []\nself.rotation_angles = np.empty(0, dtype=np.float64)\nself.reached_milestones.clear()\nself.current_state = self._simulate()\nself.prev_dist = self._distance()\nreturn self._get_obs(), self._get_info()', 'quantum_rl/2.environment.py — estratto di reset(); le funzioni self.* richiamate sono nello stesso file, np.empty appartiene a NumPy.')
p('Con _get_obs() preparo l’osservazione iniziale da dare alla rete. Con _get_info() preparo invece le informazioni leggibili, come fidelity e lista dei gate. L’osservazione serve a decidere; info serve a controllare il risultato.')
sub('Secondo passaggio: l’agente sceglie un gate')
p('Ora devo decidere quale gate aggiungere. L’osservazione è una scheda del tentativo: descrive dove sono arrivato, quali gate ho già usato e quanto sono lontano dal target. La rete legge questa scheda con forward() e assegna un punteggio alle mosse possibili. All’inizio non conosce già la soluzione: impara dopo molti tentativi.')
p('RLlib trasforma i punteggi in probabilità di scelta. Per esempio, potrebbe dare a H(0) una probabilità del 60% e dividere il resto fra altre mosse ammesse. È un esempio inventato, non un valore misurato. Durante il training posso provare anche mosse meno probabili; senza esplorazione scelgo quella più probabile.')
p('Suppongo venga scelto H(0), cioè Hadamard sul qubit zero. È il primo dei due qubit: la numerazione parte da zero. H crea una sovrapposizione, quindi il circuito non resta soltanto in |00⟩. Non impongo questa scelta all’agente: la uso per seguire un episodio concreto.')
p('La scelta dell’azione viene richiesta a RLlib: in 5.evaluate.py uso il suo metodo compute_single_action(). Questo metodo appartiene alla libreria, mentre il calcolo dei punteggi della mia rete è in forward().')
sub('Terzo passaggio: aggiungo il gate al circuito')
p('La scelta arriva come numero, non come testo “H(0)”. Uso un catalogo, simile a un elenco numerato: con due qubit l’indice 6 corrisponde a H(0). Passo quindi 6 a step(action). La funzione controlla che la mossa sia ammessa, legge la voce 6 e la aggiunge a current_gates. Questa lista è la ricetta del circuito: per ora contiene soltanto H(0).')
p('Se scegliessi RX, RY o RZ, dovrei decidere anche quanto ruotare il qubit. L’agente sceglie il tipo di rotazione; il programma aggiunge un angolo zero e lo regola con Adam. Per H e CNOT non devo cercare un angolo.')
code('gate = self.actions[action]\nself.current_gates.append(gate)\nif gate.name in ROTATIONS:\n    self.rotation_angles = np.append(self.rotation_angles, 0.0)', 'quantum_rl/2.environment.py — estratto di step(); append aggiunge alla lista e np.append appartiene a NumPy.')
p('Se ci sono angoli, step(action) richiama _optimise_angles() e confronta i tentativi descritti nella sezione 6. In questo esempio ho soltanto H, quindi salto l’ottimizzazione e passo alla simulazione.')
sub('Quarto passaggio: calcolo cosa ha prodotto il circuito')
p('Con _simulate() eseguo virtualmente la ricetta e vedo il risultato. Dopo H(0) ho (|00⟩ + |10⟩)/√2. Se misurassi i qubit, potrei leggere 00 oppure 10, ciascuno con probabilità del 50%. Bell richiede invece 00 oppure 11, con le corrette ampiezze: manca ancora il collegamento tra i qubit.')
p('Con _distance() confronto lo stato completo con Bell, tenendo conto anche delle ampiezze complesse e non soltanto delle probabilità di misura. Ottengo F=0,25, quindi d=1−0,25=0,75. Sono momentaneamente più lontano dal target, anche se H servirà per completare il circuito.')
p('A questo punto step(action) confronta il nuovo errore con quello precedente e calcola la reward. In questa prima mossa l’errore aumenta, quindi il premio è negativo. Preparo poi la nuova osservazione con _get_obs() e la restituisco all’agente.')
sub('Quinto passaggio: uso il risultato per la scelta successiva')
p('L’agente legge la scheda aggiornata e suppongo scelga CNOT(0,1). Il qubit zero controlla il qubit uno: quando il controllo vale 1, il target viene invertito. Quindi |00⟩ resta com’è e |10⟩ diventa |11⟩. Ottengo (|00⟩ + |11⟩)/√2: è Bell.')
p('Passo a step(action) l’indice 8, che a due qubit identifica questo CNOT. La ricetta contiene ora H(0) e CNOT(0,1). La risimulo e ottengo fidelity idealmente 1. Assegno i bonus di successo e chiudo il tentativo: non serve aggiungere altri gate.')
code('self.current_state = self._simulate()\ndistance = self._distance()\n# Qui calcolo la reward, descritta nella sezione 5.\nterminated = distance < self.threshold\ntruncated = len(self.current_gates) >= self.max_gates', 'quantum_rl/2.environment.py — passaggi di step(); _simulate() e _distance() sono definiti nello stesso script.')
p('terminated segnala il successo; truncated segnala che ho esaurito il budget. Se non raggiungo nessuna delle due condizioni, l’agente riceve l’osservazione aggiornata e sceglie un altro gate. PPO raccoglie queste mosse e le reward e, dopo aver raccolto un gruppo di esperienze, aggiorna i pesi della rete.')
sub('4.3 Simulazione e osservazione')
p('Il simulatore è il banco di prova: gli consegno la ricetta e lui calcola lo stato prodotto. _simulate() passa gli angoli a _state_torch(), che legge current_gates. Se trova H e CNOT, parte da |00⟩ e applica prima H, poi CNOT. A ogni prova riparte dall’inizio, senza riapplicare H allo stato finale precedente. Uso il simulatore PennyLane default.qubit.')
code('for gate in gates:\n    if gate.name == "CNOT":\n        qml.CNOT(wires=gate.wires)\n    elif gate.name == "H":\n        qml.Hadamard(wires=gate.wires[0])\n    else:\n        getattr(qml, gate.name)(parameters[parameter_index],\n                                wires=gate.wires[0])\n        parameter_index += 1\nreturn qml.state()', 'quantum_rl/2.environment.py — estratto del circuito interno a _state_torch(); le operazioni qml.* appartengono a PennyLane.')
p('Il ciclo legge un’istruzione della ricetta alla volta. Per H e CNOT prende i qubit indicati; per una rotazione prende anche l’angolo. Alla fine restituisce le ampiezze, cioè i numeri che descrivono lo stato. Durante la ricerca degli angoli tengo anche le informazioni necessarie per capire come cambiarli. Per una semplice verifica le disattivo con torch.no_grad(), una funzione PyTorch usata in _simulate() in quantum_rl/2.environment.py.')
p('Con _get_obs() preparo la scheda da dare alla rete. Deve contenere numeri, perché la rete non legge la ricetta a parole. Inserisco le ampiezze, separando parte reale e immaginaria, la quota di gate già usati, l’errore e la storia delle scelte con gli angoli. Le caselle della storia ancora vuote valgono zero.')
code('osservazione = [Re(ψ), Im(ψ), gate_usati / max_gates, d, storia]\nDimensione = 2·2ⁿ + 2 + 2·max_gates')
p('Ho incluso la storia perché due circuiti possono produrre lo stesso stato ma avere parametri diversi da riottimizzare. L’osservazione contiene anche action_mask, che indica quali azioni posso scegliere. Uso lo statevector completo perché lavoro in simulazione.')
p('Per esempio, dopo il primo H l’osservazione indica che ho già usato un gate, che l’errore vale 0,75 e che nella storia compare H(0). Il target resta sempre lo stesso; cambiano il circuito costruito e i dati che descrivono il tentativo.')
sub('4.4 Come blocco le azioni inutili')
p('_action_mask() prepara un elenco di permessi: 1 significa “puoi scegliere questa mossa”, 0 significa “è bloccata”. All’inizio un CNOT non cambia |00⟩ perché il controllo è zero. Lo blocco finché non inserisco gate che possono attivare il controllo. Blocco anche RZ su qubit inattivi: applicato a |0⟩, da solo non crea la sovrapposizione che mi serve.')
p('Valuto l’attività guardando la ricetta. RX, RY e H rendono un qubit potenzialmente attivo, anche se un angolo attuale vale zero: potrebbe essere cambiato più avanti. Un CNOT con controllo attivo può coinvolgere anche il suo target.')
p('Blocco anche gate identici ripetuti, compresi quelli separati solo da operazioni su qubit diversi. Due H identici si annullano e due rotazioni consecutive sullo stesso asse possono essere unite. La maschera aiuta quindi l’agente a evitare sequenze ridondanti. step() verifica sempre che l’azione scelta sia ammessa.')
p('All’inizio, per esempio, CNOT(0,1) è bloccato perché il controllo è ancora fermo in |0⟩. Dopo H(0) il controllo è attivo e posso scegliere quel CNOT. La maschera evita quindi una prima mossa senza effetto, lasciando disponibile il gate quando può servire.')
h('5. Calcolo della reward')
p('Dopo ogni gate devo dire all’agente se la scelta è stata utile. Per questo calcolo la reward dentro step(). È un punteggio: non modifica lo stato quantistico e non ottimizza gli angoli. PPO la usa per imparare quali scelte conviene ripetere.')
p('Per calcolarla mi servono l’errore prima del gate, salvato in prev_dist, e quello dopo il gate e l’eventuale ottimizzazione, salvato in distance. A questi aggiungo il costo del gate e gli eventuali premi. Tutti i valori seguenti usano i parametri standard del codice.')
code('reward = 10 · ln((d_precedente + ε) / (d_attuale + ε)) − 1\n         + bonus_soglia + bonus_finale\nε = 10⁻¹²')
sub('5.1 Miglioramento e costo del gate')
p('Il primo termine è positivo se l’errore diminuisce e negativo se aumenta. Il logaritmo premia il miglioramento relativo: dimezzare l’errore dà circa 6,93 punti. Ogni gate costa un punto, anche se non cambia lo stato. ε evita di calcolare il logaritmo di zero.')
p('Se passo da fidelity 0,80 a 0,90, l’infedeltà scende da 0,20 a 0,10: ho dimezzato l’errore e ricevo circa 5,93 punti dopo il costo del gate. Lo stesso premio vale passando da errore 0,10 a 0,05. Non premio quindi soltanto la differenza numerica di fidelity, ma quanto ho ridotto l’errore rispetto a prima.')
table(['Cambiamento dell’errore','Reward prima degli altri bonus'],[['Da 0,2 a 0,1','10 ln(2) − 1 ≈ +5,93'],['Errore invariato','−1'],['Da 0,1 a 0,2','10 ln(0,5) − 1 ≈ −7,93']])
code('epsilon = 1e-12\nreward = (\n    -math.log(distance + epsilon)\n    + math.log(self.prev_dist + epsilon)\n) * self.loss_mult - self.step_penalty\nself.prev_dist = distance', 'quantum_rl/2.environment.py — estratto di step(); math.log è il logaritmo della libreria standard Python.')
p('Nel codice loss_mult vale 10 e step_penalty vale 1. Aggiorno prev_dist soltanto dopo il confronto: così, al prossimo gate, il punto di partenza sarà l’errore appena ottenuto. Se lo aggiornassi prima, confronterei lo stesso valore con sé stesso e perderei il segnale di miglioramento.')
sub('5.2 Premio per le soglie')
p('Quando l’infedeltà scende sotto una soglia prevista, aggiungo dieci punti. Uso reached_milestones per assegnare il premio una sola volta per episodio. Con la soglia standard 0,01 c’è un solo traguardo, coincidente con il successo. Soglie più strette possono aggiungere altri traguardi.')
code('for exponent in range(2, int(-math.log10(self.threshold)) + 1, 2):\n    milestone = 10 ** (-exponent)\n    if distance < milestone and milestone not in self.reached_milestones:\n        reward += self.bonus_milestone\n        self.reached_milestones.add(milestone)', 'quantum_rl/2.environment.py — estratto di step(); range e int sono funzioni Python, math.log10 è della libreria standard.')
p('Con threshold=0,01 il ciclo controlla soltanto 10⁻²: non assegna bonus a 0,1 o a 0,05. Il miglioramento fino alla soglia è già premiato dal logaritmo. I traguardi raggiunti vengono cancellati da reset() all’inizio del tentativo successivo.')
sub('5.3 Bonus finale e numero di gate')
p('Quando raggiungo il target aggiungo cento punti e cinque punti per ogni gate disponibile ma non usato. Se il limite è trenta e risolvo il problema con tre gate, questo secondo premio vale 5·27 = 135 punti.')
code('gates_saved = self.max_gates - len(self.current_gates)\nbase_bonus = gates_saved * 5\nreward += self.final_rew + base_bonus + exp_bonus', 'quantum_rl/2.environment.py — estratto del ramo di successo di step(); len è una funzione Python.')
p('exp_bonus confronta la lunghezza della soluzione con il miglior risultato precedente dello stesso ambiente. Al primo successo vale zero. In seguito vale cinquanta punti se uso lo stesso numero di gate del record, aumenta se ne uso meno e diminuisce se ne uso di più.')
code('exp_bonus = 50 · exp(−0,5 · max(L − L_best, −2))')
p('L è la lunghezza attuale e L_best il record precedente. Il limite −2 evita una crescita eccessiva del bonus. Il record viene aggiornato dopo il premio e rimane tra gli episodi dello stesso ambiente. In valutazione creo invece un ambiente nuovo per ogni episodio.')
table(['Soluzione rispetto al record','exp_bonus'],[['Primo successo nell’ambiente','0'],['Un gate in più','Circa 30,33'],['Stesso numero di gate','50'],['Un gate in meno','Circa 82,44'],['Almeno due gate in meno','Circa 135,91']])
p('Questi bonus arrivano soltanto al successo. Se finisco il budget senza raggiungere il target, non ricevo né i cento punti né i premi di brevità. Resta la somma delle reward ottenute durante le mosse.')
sub('5.4 Un esempio completo di calcolo')
p('Considero un passaggio ipotetico da errore 0,020 a 0,005, con un circuito che risolve il target al terzo gate su un limite di sei. Suppongo che sia il primo successo di questo ambiente, quindi il bonus del record è zero. Non è un nuovo risultato sperimentale: serve solo a mostrare i conti.')
table(['Parte del punteggio','Calcolo','Punti'],[['Miglioramento','10 ln(0,020 / 0,005)','+13,86'],['Costo del gate','−1','−1'],['Soglia superata','0,005 < 0,01','+10'],['Successo','Bonus fisso','+100'],['Gate risparmiati','5·(6−3)','+15'],['Record','Primo successo','0'],['Reward della mossa','Somma dei contributi','Circa 137,86']])
p('Se, nello stesso ambiente, esistesse già un record di tre gate, si aggiungerebbero cinquanta punti. L’esempio mostra perché la reward finale non dipende soltanto dalla fidelity: contano anche lunghezza e successi precedenti. Il ritorno dell’episodio è invece la somma dei punteggi di tutte le mosse.')
sub('5.5 Perché considero anche il premio futuro')
p('Riprendo Bell: prima di H(0) l’errore è 0,5; dopo H(0) è 0,75. La reward è 10 ln(0,5/0,75) − 1 ≈ −5,05, senza bonus. Dopo CNOT(0,1), però, raggiungo Bell e ricevo il premio per il forte miglioramento e i bonus finali. L’agente deve imparare ad accettare un primo passo negativo quando permette di completare il circuito.')
p('La reward favorisce questa ricerca, ma dipende anche dal budget e dai bonus. Per confrontare gli esperimenti guardo soprattutto fidelity, successi e gate. La baseline Qiskit usa una reward diversa, quindi i suoi punteggi non hanno la stessa scala di quelli del training.')
h('6. Ottimizzazione degli angoli')
p('Se l’agente sceglie una rotazione, devo trovare anche il suo angolo. Per farlo step(action) richiama _optimise_angles(). Ottimizzo tutti gli angoli già presenti, non soltanto quello nuovo, mantenendo fissa la sequenza dei gate.')
p('In _optimise_angles() tratto gli angoli come manopole da regolare. Tengo fissa la ricetta, provo gli angoli attuali e misuro l’errore. Il gradiente indica in quale direzione conviene girare ciascuna manopola; Adam decide il piccolo aggiornamento da fare. Simulo di nuovo e ripeto. Solo alla fine scelgo il risultato migliore e calcolo la reward.')
code('optimiser = torch.optim.Adam([parameters], lr=0.08)\n# Schema del ciclo interno:\noptimiser.zero_grad()\nstate = self._state_torch(parameters)\nfidelity = torch.abs(torch.sum(torch.conj(target) * state)) ** 2\nloss = 1.0 - fidelity.real\nloss.backward()\noptimiser.step()', 'quantum_rl/2.environment.py — schema di _optimise_angles(); _state_torch() è nello stesso file, Adam e le operazioni sui tensori sono di PyTorch.')
p('Il limite standard è sessanta passi per tentativo. Conservo gli angoli migliori incontrati, perché l’ultimo aggiornamento non è sempre il migliore. Interrompo prima se l’errore è abbastanza basso, il gradiente è quasi nullo o compaiono valori non validi.')
p('In step() provo prima gli angoli correnti. Se non raggiungo il target, provo una seconda partenza casuale tra −π e π. Confronto le fidelity e tengo il risultato migliore. Questa procedura, chiamata multi-start, aiuta a uscire da punti iniziali poco utili. Nel codice sono previsti al massimo due tentativi.')
p('Ripeto l’ottimizzazione anche dopo un CNOT, se ci sono rotazioni nel circuito. Il CNOT non ha angoli, ma può rendere utili quelli dei gate precedenti. Se non ci sono rotazioni, salto Adam e simulo direttamente.')
sub('6.1 Il problema degli angoli a zero')
p('Una nuova rotazione parte da zero. Su Bell e GHZ, finché mancano i CNOT necessari, una rotazione può non migliorare la fidelity. Adam può quindi lasciarla a zero o riportarla verso zero. L’agente continua ad aggiungere gate senza ottenere un premio utile.')
p('Per GHZ servono collegamenti fra tutti e tre i qubit. Il problema è più difficile rispetto a Bell perché occorrono più scelte corrette prima del miglioramento. Ho affrontato questo blocco con partenze casuali, riottimizzazione dopo i CNOT e introduzione di H. Hadamard crea una sovrapposizione fissa, che Adam non può annullare cambiandone l’angolo.')
h('7. Transformer e PPO')
p('Ho usato un Transformer per leggere l’osservazione e proporre il prossimo gate. PPO è invece l’algoritmo che aggiorna i pesi della rete in base alle esperienze raccolte. I due componenti hanno quindi compiti diversi.')
sub('7.1 Come funziona la rete')
p('La rete è nella classe MaskedTransformerModel, in quantum_rl/3.model.py. Il metodo forward() legge la scheda del tentativo. Ogni numero viene trasformato in un gruppo di 128 valori, chiamato token: è una rappresentazione che la rete può modificare e combinare con le altre. Aggiungo anche una posizione, per distinguere un’ampiezza dall’errore o da un gate passato.')
p('I due blocchi di attention confrontano le informazioni della scheda. Per esempio, la rete può mettere in relazione la presenza di H(0), il qubit attivo e la possibilità di usare CNOT(0,1). Le otto teste di attention per blocco permettono di elaborare più relazioni. Queste relazioni vengono apprese con il training, non scritte come regole che impongono Bell.')
code('values = state_tensor.float().unsqueeze(-1)\ntokens = self.token_embedding(values) + self.position_embedding\nencoded = self.encoder(tokens).mean(dim=1)\nlogits = self.policy(encoded)\nself._value_out = self.value(encoded).squeeze(1)', 'quantum_rl/3.model.py — estratto di forward(); embedding, encoder, policy e value sono moduli PyTorch definiti nella stessa classe.')
p('Riunisco le informazioni facendo la media dei token. Da questo riassunto ottengo due risposte. La policy dice quali mosse sembrano più promettenti. Il critic stima quanti punti potrei ancora ottenere: aiuta a capire se il tentativo è andato meglio o peggio del previsto. Prima della scelta assegno −10⁹ alle mosse vietate dalla maschera, così non vengono selezionate.')
p('Per esempio, dopo H(0) la rete può assegnare un punteggio alto a CNOT(0,1). RLlib converte i punteggi in una scelta. Il metodo value_function() restituisce invece la stima del critic: serve a PPO per valutare le esperienze, non per applicare un gate.')
sub('7.2 Come aggiorno la policy')
p('PPO impara dai tentativi. Conservo la situazione di partenza, la mossa fatta e i punti ricevuti. Dopo molte mosse, confronto i risultati con le aspettative del critic. Se una scelta porta a risultati migliori del previsto, PPO tende a renderla più probabile; se va peggio, tende a ridurla. Gli aggiornamenti sono limitati per non cambiare strategia troppo bruscamente. Uso PPO di RLlib [1].')
p('Nel caso di Bell, PPO vede anche il successo arrivato dopo la coppia H e CNOT. Può quindi imparare il valore del primo H pur avendo ricevuto una reward negativa per quella singola mossa. Gli angoli vengono adattati dentro l’ambiente; i pesi del Transformer vengono invece aggiornati da PPO usando gruppi di esperienze, nel training avviato da 2.train.py.')
table(['Parametro','Valore'],[['Dimensione interna del Transformer','128'],['Blocchi / teste di attention','2 / 8'],['Learning rate PPO','0,0003'],['Batch di training','512 step'],['Minibatch','64 step'],['Epoche per aggiornamento','5'],['Coefficiente di entropia','0,02']])
p('L’entropia incoraggia l’esplorazione durante il training. In valutazione uso explore=False. Gli angoli possono comunque cambiare per le ripartenze casuali di Adam. Addestro la policy su un target alla volta: questi esperimenti non misurano l’uso della stessa rete su target nuovi.')
h('8. Training e valutazione')
sub('8.1 I livelli dell’addestramento')
table(['Termine','Significato'],[['Step','Scelgo e aggiungo un gate.'],['Episodio','Provo a costruire un circuito, dall’identità fino al successo o al limite di gate.'],['Iterazione PPO','Raccolgo dati da più episodi e aggiorno la rete.'],['Run','Eseguo un intero addestramento con una nuova rete iniziale.']])
p('2.train.py legge target, budget, iterazioni e worker dalla riga di comando. Registra ambiente e modello in RLlib e avvia Ray. A ogni iterazione salva nel CSV reward media, lunghezza media degli episodi e step raccolti.')
p('Ogni venti iterazioni, e alla fine, chiamo evaluate_best_circuit() per valutare la policy e save_complete_checkpoint() per salvare il checkpoint. Il salvataggio passa da una cartella temporanea e verifica la presenza dei file principali. Salvo anche il miglior circuito osservato nelle valutazioni periodiche. La scelta privilegia la fidelity e, in caso di parità, il numero minore di gate.')
sub('8.2 Ripetizione degli esperimenti')
p('3.run_repeated_training.py avvia prima il training e poi la valutazione. Ripete questa procedura per il numero di run richiesto. Ogni run ha una cartella run_i con checkpoint, cronologia e report JSON. Le run partono una dopo l’altra; il parallelismo riguarda i worker dentro ciascun training.')
p('La procedura di una singola run è in run_training_once(): avvia 2.train.py, attende che finisca e lancia 5.evaluate.py sul checkpoint finale.')
p('Nel riepilogo final_fidelity e gate_count descrivono il miglior episodio della run. total_reward è invece la reward media degli episodi valutati. success indica se il miglior episodio supera la soglia. Il tempo comprende training e valutazione finale.')
sub('8.3 Valutazione e grafici')
p('5.evaluate.py ricarica soltanto la policy dal checkpoint e controlla che target e dimensioni siano compatibili. Esegue più episodi e salva fidelity media e migliore, reward media, gate medi e percentuale di successi.')
p('Prima del caricamento uso validate_checkpoint() per controllare i file del checkpoint. Nel ciclo di valutazione richiamo poi reset() e step(action), cioè le stesse funzioni dell’ambiente descritte nella sezione 4.')
p('Con --random posso usare lo stesso ambiente scegliendo a caso tra le azioni ammesse. Questa modalità mantiene rotazioni e ottimizzazione degli angoli: è utile per un confronto con l’agente a parità di strumenti. La baseline di 1.pre_trained_table.py usa invece Qiskit e i gate H, X, Y, Z e CX, senza angoli da ottimizzare.')
p('La baseline Qiskit controlla la fidelity dopo ogni gate e si ferma al successo o al limite. Usa reward = 10F − 0,1L. Con questi gate non può preparare un generico stato arbitrario, quindi considero questo limite quando confronto i target random.')
p('La costruzione di un singolo circuito casuale è in run_random_circuit(): aggiunge un gate alla volta e restituisce esito, fidelity, numero di gate, tempo e sequenza trovata.')
p('4.plot_training_summary.py legge i CSV e crea due grafici: reward media rispetto alle iterazioni e fidelity finale rispetto alla baseline. Prima rimuove la riga Media, per non contarla come una run aggiuntiva.')
p('Per leggere i dati uso load_summary(), che rimuove la riga finale di media, e load_training_history(), che controlla e ordina le iterazioni della singola run.')
h('9. Esecuzione sul cluster HPC')
p('Ho usato il cluster per eseguire più addestramenti, anche lunghi. SLURM assegna le risorse e avvia job_train.slurm. Il job attiva l’ambiente Conda quantum-rl e lancia lo script delle run ripetute.')
code('''# Risorse indicate in job_train.slurm
#SBATCH --nodes=1
#SBATCH --partition=cpu
#SBATCH --mem=16G
#SBATCH --time=1-00:00:00
#SBATCH --ntasks-per-node=6

# Comando Python del job
python 3.run_repeated_training.py --target custom \
  --target-file targets/random_target_3_qubit_normalized.json \
  --qubits 3 --max-gates 30 --iterations 10 \
  --workers 2 --episodes 10 --runs 10 \
  --checkpoint-root checkpoints_random3 --output training_random3.csv''')
p('Ray usa due worker per raccogliere esperienze. Ogni worker simula i propri episodi; PPO usa poi i dati per aggiornare la rete. Nel codice imposto un thread PyTorch per worker, così evito che ogni processo provi a usare tutti i core del nodo.')
p('Il costo maggiore nasce dalle simulazioni ripetute e dall’ottimizzazione degli angoli a ogni step. Con n qubit servono 2ⁿ ampiezze: aumentando i qubit, memoria e lavoro crescono rapidamente. Più passi Adam possono migliorare gli angoli, ma rendono ogni decisione più lenta. Anche aumentare le iterazioni PPO richiede più tempo.')
p('Per inviare un esperimento uso sbatch job_train.slurm; per controllare la coda uso squeue. Il job può continuare sul cluster anche dopo che ho spento il computer locale. Nei miei esperimenti uso CPU e un solo nodo.')
h('10. Risultati')
p('Riporto i risultati dei miei esperimenti. Ogni campagna comprende dieci run. Presento prima le quattro baseline, poi i training con i rispettivi grafici.')
summary=[]
result_notes = {
    'Stato target: Bell': 'Con i circuiti casuali ho ottenuto sette successi su dieci. La fidelity media è 0,7750 e il numero medio di gate è 18,1. Il riferimento breve è H(0) seguito da CNOT(0,1).',
    'Stato target: GHZ': 'Per GHZ ho ottenuto cinque successi su dieci, con fidelity media 0,6000 e 24,5 gate medi. Un circuito di riferimento usa H e due CNOT.',
    'Stato target: Random a 2 qubit': 'Non ho ottenuto successi: tutti i tentativi arrivano a trenta gate e la fidelity media è 0,2174. I gate della baseline non permettono di preparare un generico stato arbitrario.',
    'Stato target: Random a 3 qubit': 'Anche a tre qubit la baseline non raggiunge il target. La fidelity media è 0,0923 e tutti i tentativi usano trenta gate.',
    'Bell: successo nell’80% delle run': 'Su Bell ho ottenuto otto successi su dieci, con fidelity media 0,8970. Le soluzioni riuscite usano tre gate; le due run fallite restano a 0,5. Il tempo totale è circa un’ora e trentadue minuti.',
    'GHZ: il primo training si blocca': 'Il primo training GHZ non riesce in nessuna run. La fidelity resta a 0,5 e i tentativi arrivano a trenta gate. Il tempo totale è circa un’ora e quarantasei minuti.',
    'GHZ: da 0 a 8 run riuscite su 10': 'Dopo le modifiche ho ottenuto otto successi su dieci e fidelity media 0,8999. Diverse run trovano H(1) → CNOT(1,2) → CNOT(2,0), con fidelity 1. Il tempo totale è circa quattro ore e sedici minuti.',
    'Random a 2 qubit: il 50% delle run riesce': 'Ho ottenuto cinque successi su dieci e fidelity media 0,8409. La soluzione riuscita più corta usa sei gate con F≈0,9907. Il tempo totale è circa cinque ore e trenta minuti.',
    'Random a 3 qubit: miglioramento senza successo': 'Non ho ottenuto successi, ma la fidelity media sale a 0,4288 e il miglior tentativo arriva a 0,7221. Tutti i circuiti usano trenta gate. Il tempo totale è circa dodici ore e trentanove minuti.',
}
result_index=0
result_started=False
for slide in slides:
    mean=re.search(r'<tr class="mean-row">(.*?)</tr>',slide,re.S)
    images=re.findall(r'<img[^>]+src="([^"]+)"',slide,re.S)
    if not mean and not images:
        if result_started and 'data-title="Diagnosi del plateau"' in slide:
            sub('Perché GHZ si blocca')
            p('Se le rotazioni restano a zero, non creo la sovrapposizione necessaria. I CNOT da soli non bastano a partire da |000⟩. L’agente può così aggiungere gate senza cambiare il risultato, come spiegato nella sezione 6.')
        elif result_started and 'data-title="Correzioni e multi-start"' in slide:
            sub('Come ho affrontato il blocco')
            p('Ho introdotto H per mantenere una sovrapposizione fissa, provato partenze diverse per gli angoli e riottimizzato le rotazioni dopo i CNOT. La maschera aiuta inoltre a evitare azioni inutili.')
        continue
    title_match=re.search(r'<h2[^>]*>(.*?)</h2>',slide,re.S)
    title=clean(title_match.group(1)) if title_match else 'Risultati'
    if mean:
        result_started=True
        result_index+=1
        if result_index == 1:
            parts.append('<h3 class="training-phase">PRIMA DEL TRAINING</h3>')
            p('Nella baseline casuale è disponibile anche il gate H. Senza H, usando soltanto X, Y, Z e CNOT, non potrei creare la sovrapposizione necessaria per Bell e GHZ: i successi su questi target sarebbero zero.')
        elif result_index == 5:
            parts.append('<h3 class="training-phase">DOPO IL TRAINING</h3>')
            p('Nel primo training H non è disponibile: posso ottenere il target usando le rotazioni con angoli ottimizzati e i CNOT. Nel training GHZ aggiornato introduco anche H, per aiutare l’agente a superare il blocco descritto più avanti.')
        sub('10.'+str(result_index)+' '+title)
        if title in result_notes: p(result_notes[title])
        rows=[]
        for row in re.findall(r'<tr[^>]*>(.*?)</tr>',slide,re.S):
            cells=[clean(x) for x in re.findall(r'<td[^>]*>(.*?)</td>',row,re.S)]
            if cells:
                cells=[html.escape(c.replace('✅ ', '').replace('❌ ', '')) for c in cells]
                rows.append(cells)
        parts.append('<div class="result-table">')
        table(['Run','Esito','Fidelity','Gate','Reward','Circuito trovato'],rows)
        parts.append('</div>')
        values=[clean(x) for x in re.findall(r'<td[^>]*>(.*?)</td>',mean.group(1),re.S)]
        summary.append([title,values[1],values[2],values[3],values[4]])
    if images and result_started:
        for image_index, src in enumerate(images):
            if src.startswith('plots/') and (ROOT/src).is_file(): src=(ROOT/src).as_uri()
            if src.startswith(('data:image/','file:')):
                description = 'Confronto della fidelity' if image_index == 0 else 'Reward durante il training'
                compact = ' final-target' if 'random3/' in src else ''
                parts.append('<figure class="result-figure'+compact+'"><img src="'+src+'"><figcaption>'+title+' — '+description+'.</figcaption></figure>')
h('11. Conclusioni e possibili miglioramenti')
p('Con questo progetto ho combinato la scelta dei gate tramite PPO con l’ottimizzazione degli angoli tramite Adam. La divisione dei compiti mi permette di cercare sia la struttura del circuito sia i parametri delle rotazioni. La reward guida l’agente verso alta fidelity e pochi gate.')
p('Ho ottenuto buoni risultati su Bell e GHZ. Il target random a due qubit è riuscito in metà delle run; quello a tre qubit resta il caso più difficile. Il passaggio di GHZ da zero a otto successi mostra quanto contino la scelta dei gate e la gestione degli angoli.')
p('Come sviluppo, aumenterei le iterazioni e confronterei una modifica alla volta: numero di tentativi Adam, presenza di H e maschera delle azioni. Userei anche la policy casuale nello stesso ambiente per misurare meglio il contributo di PPO. Più calcolo può aiutare, ma non garantisce risultati migliori.')
p('Per rendere gli esperimenti più ripetibili, fisserei i seed e salverei insieme versioni delle librerie, comandi e metriche. Per studiare meglio la parte HPC, confronterei gli stessi esperimenti con diversi numeri di worker. Infine, potrei introdurre rumore e limiti sui collegamenti tra qubit per avvicinarmi ai dispositivi reali.')
h('Riferimenti')
p('[1] J. Schulman et al., <i>Proximal Policy Optimization Algorithms</i>, 2017. <a href="https://arxiv.org/abs/1707.06347">arxiv.org/abs/1707.06347</a>.')
p('[2] A. Vaswani et al., <i>Attention Is All You Need</i>, 2017. <a href="https://arxiv.org/abs/1706.03762">arxiv.org/abs/1706.03762</a>. Riferimento per l’architettura Transformer.')
style='''@page{size:A4;margin:20mm 19mm;@bottom-right{content:counter(page);font-family:Arial;font-size:9pt;color:#526571}}*{box-sizing:border-box}body{font-family:Cambria,"Times New Roman",serif;font-size:11pt;line-height:1.45;color:#17202a;margin:0}p{text-align:justify;margin:0 0 9pt;orphans:3;widows:3}h1,h2,h3{font-family:"Segoe UI",Arial,sans-serif;color:#173956}h1{font-size:28pt;line-height:1.2}h2{font-size:17pt;margin:23pt 0 11pt;break-after:avoid}h3{font-size:12pt;margin:15pt 0 7pt;break-after:avoid}pre{font-family:Consolas,monospace;font-size:8.5pt;line-height:1.4;white-space:pre-wrap;overflow-wrap:anywhere;background:#f2f5f7;border-left:3px solid #32728d;padding:9pt;break-inside:avoid}table{border-collapse:collapse;width:100%;font-size:9pt;margin:9pt 0 14pt}td,th{padding:5pt 6pt;border:1px solid #ccd6dd;text-align:left;vertical-align:top}th{background:#eaf0f4;font-family:"Segoe UI",Arial,sans-serif}tr{break-inside:avoid}thead{display:table-header-group}.cover{height:248mm;display:flex;flex-direction:column;justify-content:center;text-align:center;break-after:page}.university{font-family:"Segoe UI";font-size:13pt;line-height:1.7;margin-bottom:25mm}.subtitle{font-size:17pt;margin:12mm 0}.author{font-size:19pt;margin:15mm 0}nav{break-after:page}nav li{margin:6pt 0}figure{margin:14pt 0;break-inside:avoid}figure img{max-width:100%;max-height:105mm;display:block;margin:0 auto 8pt}.pair{break-before:page}figcaption{font-size:10pt;color:#4e616e;text-align:center}.run-table{break-inside:avoid}a{color:#1c5f80;text-decoration:none}'''
# Keep text and tables monochrome; preserve the original figure colours.
style = re.sub(r'#[0-9a-fA-F]{6}', lambda m: {
    '#17202a': '#000000', '#173956': '#000000', '#526571': '#000000',
    '#32728d': '#000000', '#4e616e': '#000000', '#1c5f80': '#000000',
    '#f2f5f7': '#f2f2f2', '#ccd6dd': '#cccccc', '#eaf0f4': '#eeeeee',
}.get(m.group(0).lower(), '#000000'), style)
style += '.chapter{break-before:page;margin-top:0}.code-source{font-size:9pt;text-align:left;break-after:avoid}.result-table table{font-size:8pt;table-layout:fixed}.result-table th:nth-child(1){width:6%}.result-table th:nth-child(2){width:13%}.result-table th:nth-child(3){width:11%}.result-table th:nth-child(4){width:7%}.result-table th:nth-child(5){width:11%}.result-table th:nth-child(6){width:52%}.result-table td{overflow-wrap:anywhere}.student-id{display:inline-block;margin-top:8pt;font-size:13pt;color:#000000}'
style += '.result-figure{break-before:auto;break-inside:avoid;margin:9pt 0}.result-figure img{max-height:95mm;margin-bottom:4pt}.result-figure figcaption{font-size:9pt}.result-table table{margin-bottom:9pt}.result-table td,.result-table th{padding:4pt 5pt}.training-phase{font-size:14pt;margin-top:18pt;break-after:avoid}'
style += '.final-target img{max-height:78mm}'
document='<!doctype html><html lang="it"><meta charset="utf-8"><title>Relazione — Sintesi di circuiti quantistici</title><style>'+style+'</style><body>'+''.join(parts)+'</body></html>'
(OUT/'Relazione_tecnica.html').write_text(document,encoding='utf-8')
(OUT/'statistiche_presentazione.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print('Relazione riscritta:',len(re.sub('<[^>]+>',' ',document).split()),'parole circa; campagne:',len(summary))
