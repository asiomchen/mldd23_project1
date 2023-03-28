# Generowanie związków w oparciu o optymalny zestaw podstruktur
## Autorzy: Mateusz Iwan, Hubert Rybka, Anton Siomchen
## 1.  Wstępna lista źródeł danych, narzędzi, dodatkowych paczek Pythona
### 1.1. Źródła danych
[Creating the New from the Old: Combinatorial Libraries Generation with Machine-Learning-Based Compound Structure Optimization](https://pubs.acs.org/doi/10.1021/acs.jcim.6b00426)
### 1.2 Paczki Pythona:
* chemionforamtyczne: RDKit, OpenBabel, Pybel, deepchem
* uczenia maszynowego: scikit-learn, tensorflow, keras, pytorch, pytorch-lightning
* ogólne: numpy, pandas, matplotlib, seaborn, tqdm, jupyter, ipython
* inne: networkx....
## 2. lista potencjalnych problemów, które mogą wyniknąć podczas realizacji projektu
### 2.1. Problem 1 - generowane związki muszą posiadać realistyczną chemicznie strukturę.
### 2.2  Problem 2 - znalezienie odpowiednich parametrów modelu, aby ten dla danego celu biologicznego wskazywał fragmenty, które z punktu widzenia chemicznego rzeczywiście odpowiadają za najważniejsze oddziaływania liganda z celem.
### 2.3  Peoblem 3 - zapewnienie, aby generowane związki nie zawierały połączeń atomów o których wiadomo, że nie są stabilne w warunkach fizjologicznych.
### 2.3. Problem 4 - zapewnienie, aby generowane związki były możliwe do otrzymania na drodze nieskomplikowanej syntezy chemicznej.
### 2.4. Problem 5 - wybór generowanych związków pod kątem ich "drug-likeness" i/lub toksyczności.

## 3.  dodatkowe informacje
TBE
