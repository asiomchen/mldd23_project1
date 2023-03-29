# Generowanie związków w oparciu o optymalny zestaw podstruktur
## Autorzy: Mateusz Iwan, Hubert Rybka, Anton Siomchen
## 1.  Wstępna lista źródeł danych, narzędzi, dodatkowych paczek Pythona
### 1.1. Źródła danych
* [Creating the New from the Old: Combinatorial Libraries Generation with Machine-Learning-Based Compound Structure Optimization](https://pubs.acs.org/doi/10.1021/acs.jcim.6b00426)
* [AutoGrow4](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-020-00429-4)
* [DRlinker](https://pubs.acs.org/doi/full/10.1021/acs.jcim.2c00982)
* [Link-INVENT](https://pubs.rsc.org/en/content/articlehtml/2023/dd/d2dd00115b)
### 1.2 Paczki Pythona:
* chemionforamtyczne: RDKit, OpenBabel, Pybel, deepchem
* uczenia maszynowego: scikit-learn, tensorflow, keras, pytorch, pytorch-lightning
* ogólne: numpy, pandas, matplotlib, seaborn, tqdm, jupyter, ipython
* inne: networkx....
## 2. lista potencjalnych problemów, które mogą wyniknąć podczas realizacji projektu
- Problem 1 - Generowane związki muszą posiadać realistyczną chemicznie strukturę.
- Problem 2 - Generowane związki muszą być możliwe do otrzymania na drodze nieskomplikowanej syntezy chemicznej.
- Problem 3 - Generowane związki nie mogą zawierać połączeń atomów o których wiadomo, że nie są stabilne w warunkach fizjologicznych.
- Problem 4 - Znalezienie odpowiednich parametrów modelu, aby ten dla danego celu biologicznego wskazywał fragmenty, które z punktu widzenia chemicznego rzeczywiście odpowiadają za najważniejsze oddziaływania liganda z celem.
- Problem 5 - Wybór generowanych związków pod kątem ich "drug-likeness" i/lub toksyczności.

## 3.  dodatkowe informacje
TBE
