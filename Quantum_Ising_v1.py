# Quantum_1_Ising



import csv
import itertools
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector

random.seed(39)
np.random.seed(39)


FILES = (
    Path("/data/loto7_4716_k89.csv"),
    Path("/data/loto7_4716_k89_loto_2980.csv"),
    Path("/data/loto7_4716_k89_loto_plus_1736.csv"),
)


TITLE = "Quantum Ising"
N, K, DIM = 39, 7, 64
REGULARIZATIONS = (0.001, 0.01)
MAX_ITERATIONS = 200

# Sest spinova po registru.
# Temperatura je 1 u izabranim energetskim jedinicama.
# Tacan zbir nad 64 stanja zamenjuje Metropolis uzorkovanje.
Z = 1.0 - 2.0 * (
    (np.arange(DIM)[:, None] >> np.arange(6)) & 1
)
EDGES = Z[:, :-1] * Z[:, 1:]


def check(condition, message):
    if not condition:
        raise RuntimeError(message)


def load(path):
    rows = []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for line, row in enumerate(csv.reader(stream), 1):
            try:
                values = tuple(int(value) for value in row)
            except ValueError as error:
                raise ValueError(
                    f"{path.name}: necelobrojan red {line}."
                ) from error

            check(
                len(values) == K
                and len(set(values)) == K
                and min(values) >= 1
                and max(values) <= N
                and values == tuple(sorted(values)),
                f"{path.name}: nelegalan red {line}.",
            )
            rows.append(values)

    check(len(rows) >= 5, f"{path.name}: nedovoljno kola.")
    return np.asarray(rows, dtype=int) - 1


def initial():
    return np.concatenate((
        np.zeros(6),
        np.full(6, 0.1),
        np.zeros(5),
    ))


def transition(theta):
    fields = theta[:6]
    memory = theta[6:12]
    couplings = theta[12:]

    scores = (
        (Z @ fields + EDGES @ couplings)[:, None]
        + (Z * memory) @ Z[:N].T
    )
    return softmax(scores, axis=0), scores


def log_transition(theta):
    _, scores = transition(theta)
    return scores - logsumexp(
        scores, axis=0, keepdims=True
    )


def data_loss(probabilities, counts):
    check(
        probabilities.shape == (DIM, N)
        and np.all(np.isfinite(probabilities))
        and np.all(probabilities >= 0)
        and np.allclose(
            probabilities.sum(axis=0),
            1.0,
            atol=1e-8,
            rtol=0,
        ),
        "Neispravna uslovna raspodela.",
    )

    total = counts.sum()
    previous_counts = counts.sum(axis=0)

    loss = (
        np.sum(probabilities**2 * previous_counts)
        - 2.0 * np.sum(counts * probabilities)
    ) / total + 1.0

    derivative = 2.0 * (
        probabilities * previous_counts - counts
    ) / total

    return float(loss), derivative


def objective(theta, counts, weight):
    probabilities, _ = transition(theta)
    loss, derivative = data_loss(probabilities, counts)

    score_derivative = probabilities * (
        derivative
        - np.sum(
            derivative * probabilities,
            axis=0,
            keepdims=True,
        )
    )
    summed = score_derivative.sum(axis=1)

    gradient = np.concatenate((
        Z.T @ summed,
        np.einsum(
            "op,oq,pq->q",
            score_derivative,
            Z,
            Z[:N],
        ),
        EDGES.T @ summed,
    ))

    return (
        loss + weight * np.dot(theta, theta),
        gradient + 2.0 * weight * theta,
    )


def joint(log_weights):
    logs = np.asarray(log_weights, dtype=float).copy()
    check(
        logs.ndim == 2
        and np.all(np.isfinite(logs) | np.isneginf(logs)),
        "Neispravni logaritmi raspodele.",
    )

    slots, states = logs.shape
    check(1 <= slots <= states, "Neispravne dimenzije.")

    indices = np.arange(states)
    increasing = indices[:, None] < indices[None, :]
    allowed = (
        (indices[None, :] >= np.arange(slots)[:, None])
        & (
            indices[None, :]
            <= states - slots + np.arange(slots)[:, None]
        )
    )
    logs[~allowed] = -np.inf

    forward = np.full_like(logs, -np.inf)
    backward = np.full_like(logs, -np.inf)
    forward[0] = logs[0]

    values = logs[0].copy()
    parents = np.zeros((slots, states), dtype=int)

    for slot in range(1, slots):
        forward[slot] = logs[slot] + logsumexp(
            np.where(
                increasing,
                forward[slot - 1, :, None],
                -np.inf,
            ),
            axis=0,
        )

        candidates = np.where(
            increasing,
            values[:, None],
            -np.inf,
        )
        parents[slot] = candidates.argmax(axis=0)
        values = logs[slot] + candidates[
            parents[slot], np.arange(states)
        ]

    log_partition = float(logsumexp(forward[-1]))
    check(np.isfinite(log_partition), "Nema legalne mase.")

    path = np.zeros(slots, dtype=int)
    path[-1] = int(values.argmax())

    for slot in range(slots - 1, 0, -1):
        path[slot - 1] = parents[slot, path[slot]]

    backward[-1] = 0.0

    for slot in range(slots - 2, -1, -1):
        backward[slot] = logsumexp(
            np.where(
                increasing,
                (
                    logs[slot + 1]
                    + backward[slot + 1]
                )[None, :],
                -np.inf,
            ),
            axis=1,
        )

    marginals = np.exp(
        forward + backward - log_partition
    )
    mode_log_mass = float(
        logs[np.arange(slots), path].sum()
    )

    check(
        np.all(np.diff(path) > 0)
        and mode_log_mass <= log_partition + 1e-8
        and log_partition <= 1e-8,
        "Neispravan maksimum ili particiona funkcija.",
    )
    check(
        np.allclose(
            marginals.sum(axis=1),
            1.0,
            atol=1e-8,
            rtol=0,
        ),
        "Marginalne raspodele nisu normalizovane.",
    )

    return path, marginals, log_partition, logs


def counts_for(data, end, slot):
    counts = np.zeros((DIM, N))
    np.add.at(
        counts,
        (
            data[1:end + 1, slot],
            data[:end, slot],
        ),
        1.0,
    )
    return counts


def fit(data, end, weight):
    parameters = []

    for slot in range(K):
        counts = counts_for(data, end, slot)
        result = minimize(
            objective,
            initial(),
            args=(counts, weight),
            jac=True,
            method="CG",
            options={
                "maxiter": MAX_ITERATIONS,
                "gtol": 1e-6,
            },
        )

        loss, gradient = objective(result.x, counts, weight)
        check(
            np.isfinite(loss)
            and np.all(np.isfinite(result.x))
            and np.all(np.isfinite(gradient)),
            "Obuka nije vratila konacne parametre i gradijent.",
        )

        print(
            "Registar:",
            slot + 1,
            "status:",
            result.message,
            flush=True,
        )
        parameters.append(result.x)

    return parameters


def evaluate(parameters, data, times):
    matrices = [
        log_transition(theta)
        for theta in parameters
    ]
    losses, hits = [], []

    for time in times:
        log_weights = np.array([
            matrices[slot][:N, data[time - 1, slot]]
            for slot in range(K)
        ])
        path, _, log_partition, logs = joint(log_weights)

        loss = log_partition - logs[
            np.arange(K), data[time]
        ].sum()

        check(
            not np.isnan(loss) and loss >= -1e-8,
            "Neispravan NLL.",
        )
        losses.append(loss)
        hits.append(len(set(path) & set(data[time])))

    return (
        float(np.mean(losses)),
        np.asarray(hits, dtype=int),
    )


def verify_quantum(theta, previous):
    probabilities, scores = transition(theta)
    fields = theta[:6] + theta[6:12] * Z[previous]
    terms = []

    for qubit in range(6):
        label = ["I"] * 6
        label[5 - qubit] = "Z"
        terms.append((
            "".join(label),
            -float(fields[qubit]),
        ))

    for qubit in range(5):
        label = ["I"] * 6
        label[5 - qubit] = "Z"
        label[4 - qubit] = "Z"
        terms.append((
            "".join(label),
            -float(theta[12 + qubit]),
        ))

    hamiltonian = SparsePauliOp.from_list(terms).to_matrix()

    check(
        np.allclose(
            hamiltonian.diagonal().real,
            -scores[:, previous],
            atol=1e-10,
            rtol=0,
        ),
        "Ising Hamiltonijan nije usaglasen.",
    )

    circuit = QuantumCircuit(6)
    circuit.initialize(
        np.sqrt(probabilities[:, previous]),
        range(6),
    )
    amplitudes = Statevector.from_instruction(circuit).data

    check(
        np.allclose(
            np.abs(amplitudes)**2,
            probabilities[:, previous],
            atol=1e-10,
            rtol=0,
        ),
        "Qiskit Gibbsove amplitude nisu usaglasene.",
    )


def self_check(data):
    counts = counts_for(
        data, min(8, len(data) - 1), 0
    )
    theta = initial() + 0.03 * np.sin(
        np.arange(len(initial())) + 1
    )

    verify_quantum(theta, int(data[0, 0]))
    _, gradient = objective(theta, counts, 0.001)

    for index in range(len(theta)):
        offset = np.zeros_like(theta)
        offset[index] = 1e-6

        numerical = (
            objective(theta + offset, counts, 0.001)[0]
            - objective(theta - offset, counts, 0.001)[0]
        ) / 2e-6

        check(
            np.isclose(
                numerical,
                gradient[index],
                atol=2e-5,
                rtol=2e-4,
            ),
            f"Gradijent nije potvrdjen: {index}.",
        )

    toy = softmax(
        np.sin(np.arange(15).reshape(3, 5)),
        axis=1,
    )
    path, _, log_partition, logs = joint(np.log(toy))
    scores = [
        sum(
            logs[j, combination[j]]
            for j in range(3)
        )
        for combination in itertools.combinations(range(5), 3)
    ]

    check(
        np.isclose(
            log_partition,
            logsumexp(scores),
            atol=1e-10,
            rtol=0,
        )
        and np.isclose(
            logs[np.arange(3), path].sum(),
            max(scores),
            atol=1e-10,
            rtol=0,
        ),
        "Provera globalne raspodele nije prosla.",
    )

    extreme = initial()
    extreme[:6] = 200.0
    extreme_logs = log_transition(extreme)

    check(
        np.all(np.isfinite(extreme_logs))
        and np.allclose(
            logsumexp(extreme_logs, axis=0),
            0.0,
            atol=1e-10,
            rtol=0,
        ),
        "Provera veoma malih Gibbsovih verovatnoca nije prosla.",
    )

    print("Matematicke i pocetne Qiskit provere su prosle.")


def main():
    print(TITLE)
    print("Ising Hamiltonijan i tacna Gibbsova raspodela.")
    print("Temperatura je 1 u izabranim energetskim jedinicama.")

    datasets = [
        (path, load(path))
        for path in FILES
    ]
    self_check(datasets[0][1])
    results = []

    for path, data in datasets:
        total = len(data) - 1
        train_end = int(0.70 * total)
        validation_end = int(0.85 * total)

        check(
            0 < train_end < validation_end < total,
            "Neispravna hronoloska podela.",
        )

        print()
        print(path.name, "redova:", len(data))
        best_weight, best_loss = None, np.inf

        for weight in REGULARIZATIONS:
            parameters = fit(data, train_end, weight)
            loss, _ = evaluate(
                parameters,
                data,
                range(train_end + 1, validation_end + 1),
            )
            print(
                "Regularizacija:",
                weight,
                "validacioni NLL:",
                loss,
            )

            if np.isfinite(loss) and loss < best_loss:
                best_weight, best_loss = weight, loss

        check(
            best_weight is not None,
            "Nema konacnog validacionog NLL.",
        )

        test_parameters = fit(
            data, validation_end, best_weight
        )
        test_loss, hits = evaluate(
            test_parameters,
            data,
            range(validation_end + 1, len(data)),
        )

        print("Test kola:", len(hits))
        print("Test NLL:", test_loss)
        print(
            "Test srednji broj pogodaka:",
            float(hits.mean()),
        )

        parameters = fit(data, total, best_weight)
        log_weights = []

        for slot, theta in enumerate(parameters):
            previous = int(data[-1, slot])
            verify_quantum(theta, previous)
            log_weights.append(
                log_transition(theta)[:N, previous]
            )

        log_weights = np.asarray(log_weights)
        prediction, marginals, log_partition, _ = joint(
            log_weights
        )

        print("Log legalne mase:", log_partition)

        if np.array_equal(prediction, data[-1]):
            print("PROVERA: NEXT ponavlja poslednje kolo.")

        run = 1
        for difference in np.diff(prediction):
            run = run + 1 if difference == 1 else 1
            if run >= 5:
                print(
                    "PROVERA: najmanje pet uzastopnih brojeva."
                )
                break

        results.append((
            path.name,
            prediction + 1,
            marginals,
            hits,
            np.exp(log_weights),
        ))

    for first, second in itertools.combinations(results, 2):
        if np.array_equal(first[1], second[1]):
            print("ISTI NEXT:", first[0], second[0])
            print(
                "Razlika raspodela:",
                float(np.max(np.abs(first[4] - second[4]))),
            )

    figure, axes = plt.subplots(
        3, 2, figsize=(12, 10)
    )

    for row, result in enumerate(results):
        name, prediction, marginals, hits, _ = result

        print(name)
        print("NEXT:", " ".join(map(str, prediction)))

        axes[row, 0].imshow(
            marginals,
            aspect="auto",
        )
        axes[row, 0].set_title(name)
        axes[row, 0].set_xlabel(
            "Broj - indeks 0 predstavlja broj 1"
        )
        axes[row, 1].bar(
            np.arange(8),
            np.bincount(hits, minlength=8),
        )
        axes[row, 1].set_xlabel(
            "Pogodaka na testu"
        )

    figure.suptitle(TITLE)
    figure.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()



"""
Ising Hamiltonijan i tacna Gibbsova raspodela.
Temperatura je 1 u izabranim energetskim jedinicama.
Matematicke i pocetne Qiskit provere su prosle.

loto7_4716_k89.csv redova: 4716
Registar: 1 status: Optimization terminated successfully.
Registar: 2 status: Optimization terminated successfully.
Registar: 3 status: Optimization terminated successfully.
Registar: 4 status: Optimization terminated successfully.
Registar: 5 status: Optimization terminated successfully.
Registar: 6 status: Optimization terminated successfully.
Registar: 7 status: Optimization terminated successfully.
Regularizacija: 0.001 validacioni NLL: 18.52493524775504
Registar: 1 status: Optimization terminated successfully.
Registar: 2 status: Optimization terminated successfully.
Registar: 3 status: Optimization terminated successfully.
Registar: 4 status: Optimization terminated successfully.
Registar: 5 status: Optimization terminated successfully.
Registar: 6 status: Optimization terminated successfully.
Registar: 7 status: Optimization terminated successfully.
Regularizacija: 0.01 validacioni NLL: 18.200705064230384
Registar: 1 status: Optimization terminated successfully.
Registar: 2 status: Optimization terminated successfully.
Registar: 3 status: Optimization terminated successfully.
Registar: 4 status: Optimization terminated successfully.
Registar: 5 status: Optimization terminated successfully.
Registar: 6 status: Optimization terminated successfully.
Registar: 7 status: Optimization terminated successfully.
Test kola: 708
Test NLL: 18.15947156757307
Test srednji broj pogodaka: 1.2372881355932204
Registar: 1 status: Optimization terminated successfully.
Registar: 2 status: Optimization terminated successfully.
Registar: 3 status: Optimization terminated successfully.
Registar: 4 status: Optimization terminated successfully.
Registar: 5 status: Optimization terminated successfully.
Registar: 6 status: Optimization terminated successfully.
Registar: 7 status: Optimization terminated successfully.
Log legalne mase: -4.525604012402345

loto7_4716_k89_loto_2980.csv redova: 2979
Registar: 1 status: Optimization terminated successfully.
Registar: 2 status: Optimization terminated successfully.
Registar: 3 status: Optimization terminated successfully.
Registar: 4 status: Optimization terminated successfully.
Registar: 5 status: Optimization terminated successfully.
Registar: 6 status: Optimization terminated successfully.
Registar: 7 status: Optimization terminated successfully.
Regularizacija: 0.001 validacioni NLL: 18.617206840500494
Registar: 1 status: Optimization terminated successfully.
Registar: 2 status: Optimization terminated successfully.
Registar: 3 status: Optimization terminated successfully.
Registar: 4 status: Optimization terminated successfully.
Registar: 5 status: Optimization terminated successfully.
Registar: 6 status: Optimization terminated successfully.
Registar: 7 status: Optimization terminated successfully.
Regularizacija: 0.01 validacioni NLL: 18.309757251783378
Registar: 1 status: Optimization terminated successfully.
Registar: 2 status: Optimization terminated successfully.
Registar: 3 status: Optimization terminated successfully.
Registar: 4 status: Optimization terminated successfully.
Registar: 5 status: Optimization terminated successfully.
Registar: 6 status: Optimization terminated successfully.
Registar: 7 status: Optimization terminated successfully.
Test kola: 447
Test NLL: 18.250999074196326
Test srednji broj pogodaka: 1.3064876957494407
Registar: 1 status: Optimization terminated successfully.
Registar: 2 status: Optimization terminated successfully.
Registar: 3 status: Optimization terminated successfully.
Registar: 4 status: Optimization terminated successfully.
Registar: 5 status: Optimization terminated successfully.
Registar: 6 status: Optimization terminated successfully.
Registar: 7 status: Optimization terminated successfully.
Log legalne mase: -4.624914646980297

loto7_4716_k89_loto_plus_1736.csv redova: 1735
Registar: 1 status: Optimization terminated successfully.
Registar: 2 status: Optimization terminated successfully.
Registar: 3 status: Optimization terminated successfully.
Registar: 4 status: Optimization terminated successfully.
Registar: 5 status: Optimization terminated successfully.
Registar: 6 status: Optimization terminated successfully.
Registar: 7 status: Optimization terminated successfully.
Regularizacija: 0.001 validacioni NLL: 18.410285015308368
Registar: 1 status: Optimization terminated successfully.
Registar: 2 status: Optimization terminated successfully.
Registar: 3 status: Optimization terminated successfully.
Registar: 4 status: Optimization terminated successfully.
Registar: 5 status: Optimization terminated successfully.
Registar: 6 status: Optimization terminated successfully.
Registar: 7 status: Optimization terminated successfully.
Regularizacija: 0.01 validacioni NLL: 18.213916805087514
Registar: 1 status: Optimization terminated successfully.
Registar: 2 status: Optimization terminated successfully.
Registar: 3 status: Optimization terminated successfully.
Registar: 4 status: Optimization terminated successfully.
Registar: 5 status: Optimization terminated successfully.
Registar: 6 status: Optimization terminated successfully.
Registar: 7 status: Optimization terminated successfully.
Test kola: 261
Test NLL: 18.00470643934818
Test srednji broj pogodaka: 1.1800766283524904
Registar: 1 status: Optimization terminated successfully.
Registar: 2 status: Optimization terminated successfully.
Registar: 3 status: Optimization terminated successfully.
Registar: 4 status: Optimization terminated successfully.
Registar: 5 status: Optimization terminated successfully.
Registar: 6 status: Optimization terminated successfully.
Registar: 7 status: Optimization terminated successfully.
Log legalne mase: -4.615329906492365


loto7_4716_k89.csv
NEXT: 1 x 14 y 27 z 39

loto7_4716_k89_loto_2980.csv
NEXT: 1 x 16 y 27 z 39

loto7_4716_k89_loto_plus_1736.csv
NEXT: 1 x 11 y 28 z 38
"""



"""
1. 
Ising Model	
Spinovi, međusobne interakcije, Hamiltonijan i temperaturna dinamika. 
Koristim tačan zbir nad 64 spinska stanja, uz očuvanje Hamiltonijana i temperaturne raspodele.

2. 
ODE - Ordinary Differential Equation	
Evolucija stanja preko zadate diferencijalne jednačine i početnog uslova, uz solve_ivp.	
Diferencijalna dinamika. Desnu stranu jednačine naučiti iz prethodnih kola i proveriti evoluciju raspodele. 
Otvoreni kvantni model, jer evolucija raspodele može biti neunitarna. 

3. 
Partial Differential Equations	
PDE — Heat/Diffusion
Osnovna ideja je difuzija kroz jednačinu toplote, rešavana konačnim razlikama i vremenskim koracima. 
Za Loto treba sačuvati difuzioni operator, vremensku evoluciju, početne i granične uslove, uz učenje parametara iz istorije. 
Posebno treba proveriti očuvanje ukupne verovatnoće. 
Otvoreni kvantni model, jer evolucija raspodele može biti neunitarna. 

4. 
Fourier Transform	
DFT/FFT, kompleksni koeficijenti, spektralne amplitude, faze i inverzna transformacija.	
Poseban spektralni model koji uči vremensku evoluciju komponenti raspodele. 
Fourier uči spektralne amplitude i faze. 
U Fourier kodu FFT računanje filtera, jer množenje gustih Fourier matrica može ostaviti grešku poništavanja i kod identičnog filtera. 

5, 
Quantum Mechanics	
Kompleksna talasna funkcija, Schrödingerova evolucija, potencijal, interferencija i Bornove verovatnoće.	
Kvantni dinamički model sa početnim amplitudama i parametrima naučenim iz istorije. 
Schrödingerov model uci Hermitski Hamiltonijan.



Uključene optimizacija i vizuelizacija u svaki odgovarajući model. 

Plotting and Data Visualization	
Prikaz podataka, histogrami, dijagrami i toplotne mape.	
Dijagnostika unutar prediktivnih fajlova. 
Vizuelizacija ulazi u dijagnostiku — prikaz naučene raspodele, promene kroz vreme, konvergencije i rezultata odvojenog testa.

Optimization	
Optimization ulazi kao stvarni postupak obuke. 
Gradijentni spust, promenljiv korak, konjugovane gradijente i Nelder-Mead. 
Te postupke odabrati prema ciljnoj funkciji i hronološkoj validaciji. 
Poređenje više determinističkih početnih tačaka čuva istraživanje različitih lokalnih minimuma. 
Takvo poređenje samo po sebi ne potvrđuje globalni optimum.





Fajl	                Osnovna ideja 	                                                                                Šta treba učiti iz istorije
Ising	                Spinovi, Hamiltonijan, međusobne interakcije i temperaturna raspodela.	                        Jačine interakcija, lokalna polja i njihova zavisnost od prethodnih kola.
ODE	                    Početno stanje, diferencijalna jednačina i integracija vremenske evolucije.	                    Parametri jednačine koja opisuje prelaz raspodele u sledeće kolo.
PDE — Heat/Diffusion	Difuzioni operator, konačne razlike, vremenski koraci i granični uslovi.	                    Parametri difuzije i eventualni izvorni član, uz očuvanje nenegativnosti i ukupne verovatnoće.
Fourier	                Kompleksni spektralni koeficijenti, amplitude, faze i inverzna transformacija.	                Evolucija spektralnih komponenti kroz kola.
Quantum Mechanics	    Schrödingerova evolucija, kompleksne amplitude, Hamiltonijan, interferencija i Bornovo pravilo.	Početne amplitude i parametri Hamiltonijana koji opisuju istorijske prelaze.


Za kvantnu realizaciju posebno treba očuvati razliku između difuzije i Schrödingerove evolucije. Difuzija je neunitarna. 
Njeno preslikavanje na Qiskit zahteva odgovarajuću konstrukciju. Sama zamena difuzionog operatora unitarnim kolom promenila bi metod.

Svih pet fajlova obrađuje ista tri CSV-a zasebno, sa jednom NEXT kombinacijom po CSV-u. Razlikovaće se po naučenom modelu i postupku evolucije. 



Svaki model koristi prethodno kolo kao početno stanje. 
Obuka obuhvata sve raspoložive istorijske prelaze.

- Ising koristi tačan zbir nad 64 spinska stanja.
- ODE koristi naučeni generator prenosa verovatnoće.
- PDE koristi konačne razlike, periodične granice i izvorni član koji čuva ukupnu verovatnoću.
- Fourier uči spektralne amplitude i faze. Njegov kvantni filter koristi jedan pomoćni kubit.
- Schrödingerov model uči Hermitski Hamiltonijan i čuva unitarnu evoluciju.


Optimizacija koristi konjugovane gradijente. 
Svaki fajl ima hronološku validaciju, odvojeni test, završnu obuku i grafikone.



Statička i matematička provera: 

Kod	            Nalaz

Ising	        Logaritam računati iz energije i particione funkcije. 
                Znak Hamiltonijana, Gibbsova normalizacija, gradijenti polja i interakcija, logaritamski izlaz.

ODE	            Izveo sam gradijent generatora i proverio konstrukciju prenosa. 
                Smer prenosa, zbir kolona generatora, Fréchet gradijent i konstrukcija skočnih operatora.

PDE	            Proverio sam FTCS korak, gradijent izvora i kvantni kanal. 
                Stabilnost FTCS koraka, očuvanje verovatnoće, gradijent izvora i kvantni kanal.

Fourier	        Filter  računa FFT. Time se izbegava poništavanje pri množenju gustih Fourier matrica. Logaritmi se računaju iz amplituda.
                Smer FFT/QFT, normalizacija filtera, gradijenti amplituda i faza, raspored pomoćnog kubita. 

Schrödinger     Evolucija koristi direktni expm, a gradijent odgovarajući Fréchet izvod. Logaritmi se računaju iz amplituda.
                Hermitičnost, matrična eksponencijala, kompleksni adjoint gradijent i logaritmi amplituda.


Provereni su i svi pozivi joint() — Ising, Fourier i Schrödinger verzije prosleđuju logaritme. ODE i PDE prosleđuju verovatnoće svojoj odgovarajućoj funkciji.
Hronološke granice treninga, validacije i testa pravilno su indeksirane. Završna obuka obuhvata sve istorijske prelaze.
Konvergencija optimizatora i prolazak ugrađenih provera.
"""
