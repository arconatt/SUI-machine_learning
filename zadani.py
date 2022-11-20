# %% [markdown]
# Vítejte u druhého projektu do SUI.
# V rámci projektu Vás čeká několik cvičení, v nichž budete doplňovat poměrně malé fragmenty kódu (místo je vyznačeno pomocí `None` nebo `pass`).
# Pokud se v buňce s kódem již něco nachází, využijte/neničte to.
# Buňky nerušte ani nepřidávejte.
# 
# Až budete s řešením hotovi, vyexportujte ho ("Download as") jako PDF i pythonovský skript a ty odevzdejte pojmenované názvem týmu (tj. loginem vedoucího).
# Dbejte, aby bylo v PDF všechno vidět (nezůstal kód za okrajem stránky apod.).
# 
# U všech cvičení je uveden orientační počet řádků řešení.
# Berte ho prosím opravdu jako orientační, pozornost mu věnujte, pouze pokud ho významně překračujete.

# %%
import numpy as np
import copy
import matplotlib.pyplot as plt
import scipy.stats

# %% [markdown]
# # Přípravné práce
# Prvním úkolem v tomto projektu je načíst data, s nimiž budete pracovat.
# Vybudujte jednoduchou třídu, která se umí zkonstruovat z cesty k negativním a pozitivním příkladům, a bude poskytovat:
# - pozitivní a negativní příklady (`dataset.pos`, `dataset.neg` o rozměrech [N, 7])
# - všechny příklady a odpovídající třídy (`dataset.xs` o rozměru [N, 7], `dataset.targets` o rozměru [N])
# 
# K načítání dat doporučujeme využít `np.loadtxt()`.
# Netrapte se se zapouzdřováním a gettery, berte třídu jako Plain Old Data.
# 
# Načtěte trénovací (`{positives,negatives}.trn`), validační (`{positives,negatives}.val`) a testovací (`{positives,negatives}.tst`) dataset, pojmenujte je po řadě `train_dataset`, `val_dataset` a `test_dataset`. 
# 
# **(6 řádků)** 
# 

# %%
class BinaryDataset:
    def __init__(self, pos, neg):
        self.pos = np.array(np.loadtxt(pos))
        self.neg = np.array(np.loadtxt(neg))
        self.xs = np.concatenate((self.pos, self.neg))
        target_pos = np.repeat(1,self.pos.shape[0])
        target_neg = np.repeat(0,self.neg.shape[0])
        self.targets = np.concatenate((target_pos, target_neg))

train_dataset = BinaryDataset('positives.trn', 'negatives.trn')
val_dataset = BinaryDataset('positives.val', 'negatives.val')
test_dataset = BinaryDataset('positives.tst', 'negatives.tst')

print('positives', train_dataset.pos.shape)
print('negatives', train_dataset.neg.shape)
print('xs', train_dataset.xs.shape)
print('targets', train_dataset.targets.shape)

# %% [markdown]
# V řadě následujících cvičení budete pracovat s jedním konkrétním příznakem. Naimplementujte proto funkci, která vykreslí histogram rozložení pozitivních a negativních příkladů z jedné sady. Nezapomeňte na legendu, ať je v grafu jasné, které jsou které. Funkci zavoláte dvakrát, vykreslete histogram příznaku `5` -- tzn. šestého ze sedmi -- pro trénovací a validační data
# 
# **(5 řádků)**

# %%
FOI = 5  # Feature Of Interest

def plot_data(poss, negs):
    plt.figure()
    plt.xlabel('Value')
    plt.ylabel('Number of occurences')
    
    plt.hist(poss, histtype='step', bins='auto', label='Positive')
    plt.hist(negs, histtype='step', bins='auto', label='Negative')
    plt.legend()


plot_data(train_dataset.pos[:, FOI], train_dataset.neg[:, FOI])
plot_data(val_dataset.pos[:, FOI], val_dataset.neg[:, FOI])

# %% [markdown]
# ### Evaluace klasifikátorů
# Než přistoupíte k tvorbě jednotlivých klasifikátorů, vytvořte funkci pro jejich vyhodnocování.
# Nechť se jmenuje `evaluate` a přijímá po řadě klasifikátor, pole dat (o rozměrech [N, F]) a pole tříd ([N]).
# Jejím výstupem bude _přesnost_ (accuracy), tzn. podíl správně klasifikovaných příkladů.
# 
# Předpokládejte, že klasifikátor poskytuje metodu `.prob_class_1(data)`, která vrací pole posteriorních pravděpodobností třídy 1 pro daná data.
# Evaluační funkce bude muset provést tvrdé prahování (na hodnotě 0.5) těchto pravděpodobností a srovnání získaných rozhodnutí s referenčními třídami.
# Využijte fakt, že `numpy`ovská pole lze mj. porovnávat se skalárem.
# 
# **(3 řádky)**

# %%
def evaluate(classifier, inputs, targets):
    post = np.abs(classifier.prob_class_1(inputs) - targets)
    prah = post < 0.5
    return np.mean(prah)


class Dummy:
    def prob_class_1(self, xs):
        return np.asarray([0.2, 0.7, 0.7])

print(evaluate(Dummy(), None, np.asarray([0, 0, 1])))  # should be 0.66

# %% [markdown]
# ### Baseline
# Vytvořte klasifikátor, který ignoruje vstupní data.
# Jenom v konstruktoru dostane třídu, kterou má dávat jako tip pro libovolný vstup.
# Nezapomeňte, že jeho metoda `.prob_class_1(data)` musí vracet pole správné velikosti.
# 
# **(4 řádky)**

# %%
class PriorClassifier:
    def __init__(self, cls):
        self.cls = cls
    def prob_class_1(self, data):
        return np.repeat(self.cls, data.shape[0])

baseline = PriorClassifier(0)
val_acc = evaluate(baseline, val_dataset.xs[:, FOI], val_dataset.targets)
print('Baseline val acc:', val_acc)

# %% [markdown]
# # Generativní klasifikátory
# V této  části vytvoříte dva generativní klasifikátory, oba založené na Gaussovu rozložení pravděpodobnosti.
# 
# Začněte implementací funce, která pro daná 1-D data vrátí Maximum Likelihood odhad střední hodnoty a směrodatné odchylky Gaussova rozložení, které data modeluje.
# Funkci využijte pro natrénovaní dvou modelů: pozitivních a negativních příkladů.
# Získané parametry -- tzn. střední hodnoty a směrodatné odchylky -- vypíšete.
# 
# **(1 řádek)**

# %%
def mle_gauss_1d(data):
    return np.mean(data), np.sqrt(np.mean((data - np.mean(data))**2))

mu_pos, std_pos = mle_gauss_1d(train_dataset.pos[:, FOI])
mu_neg, std_neg = mle_gauss_1d(train_dataset.neg[:, FOI])

print('Pos mean: {:.2f} std: {:.2f}'.format(mu_pos, std_pos))
print('Neg mean: {:.2f} std: {:.2f}'.format(mu_neg, std_neg))

# %% [markdown]
# Ze získaných parametrů vytvořte `scipy`ovská gaussovská rozložení `scipy.stats.norm`.
# S využitím jejich metody `.pdf()` vytvořte graf, v němž srovnáte skutečné a modelové rozložení pozitivních a negativních příkladů.
# Rozsah x-ové osy volte od -0.5 do 1.5 (využijte `np.linspace`) a u volání `plt.hist()` nezapomeňte nastavit `density=True`, aby byl histogram normalizovaný a dal se srovnávat s modelem.
# 
# **(2 + 8 řádků)**

# %%
x_size = np.linspace(-0.5, 1.5, 1000)
pdf_pos = scipy.stats.norm.pdf(x_size, loc=mu_pos, scale=std_pos)
pdf_neg = scipy.stats.norm.pdf(x_size, loc=mu_neg, scale=std_neg)
plt.hist(train_dataset.pos[:, FOI], histtype='step', label='Positive, real', density=True)
plt.plot(x_size, pdf_pos, label='Positive, model')
plt.hist(train_dataset.neg[:, FOI], histtype='step', label='Negative. real', density=True)
plt.plot(x_size, pdf_neg, label='Negative, model')
plt.legend()
plt.show()

# %% [markdown]
# Naimplementujte binární generativní klasifikátor. 
# Při konstrukci přijímá dvě rozložení poskytující metodu `.pdf()` a odpovídající apriorní pravděpodobnost tříd.
# Dbejte, aby Vám uživatel nemohl zadat neplatné apriorní pravděpodobnosti.
# Jako všechny klasifikátory v tomto projektu poskytuje metodu `prob_class_1()`.
# 
# **(9 řádků)**

# %%
class GenerativeClassifier2Class:
    def __init__(self, norm_pos, prior_pos, norm_neg, prior_neg):
        assert(prior_neg + prior_pos == 1)
        self.prior_neg = prior_neg
        self.prior_pos = prior_pos
        self.norm_neg = norm_neg
        self.norm_pos = norm_pos
        
    def prob_class_1(self, data):
        return (self.norm_pos.pdf(data) * self.prior_pos)/((self.norm_pos.pdf(data) * self.prior_pos) + (self.norm_neg.pdf(data) * self.prior_neg)) 

# %% [markdown]
# Nainstancujte dva generativní klasifikátory: jeden s rovnoměrnými priory a jeden s apriorní pravděpodobností 0.75 pro třídu 0 (negativní příklady).
# Pomocí funkce `evaluate()` vyhodnotíte jejich úspěšnost na validačních datech.
# 
# **(2 řádky)**

# %%
classifier_flat_prior = GenerativeClassifier2Class(scipy.stats.norm(loc=mu_pos, scale=std_pos), 0.5, scipy.stats.norm(loc=mu_neg, scale=std_neg), 0.5)
classifier_full_prior = GenerativeClassifier2Class(scipy.stats.norm(loc=mu_pos, scale=std_pos), 0.25, scipy.stats.norm(loc=mu_neg, scale=std_neg), 0.75)

print('flat:', evaluate(classifier_flat_prior, val_dataset.xs[:, FOI], val_dataset.targets))
print('full:', evaluate(classifier_full_prior, val_dataset.xs[:, FOI], val_dataset.targets))

# %% [markdown]
# Vykreslete průběh posteriorní pravděpodobnosti třídy 1 jako funkci příznaku 5, opět v rozsahu <-0.5; 1.5> pro oba klasifikátory.
# Do grafu zakreslete i histogramy rozložení trénovacích dat, opět s `density=True` pro zachování dynamického rozsahu.
# 
# **(8 řádků)**

# %%
x_size = np.linspace(-0.5, 1.5, 1000)
plt.xlabel('Value')
plt.ylabel('Density')
plt.plot(x_size, classifier_flat_prior.prob_class_1(x_size), label='Positive, flat post. prob.')
plt.plot(x_size, classifier_full_prior.prob_class_1(x_size), label='Negative, full post. prob.')
plt.hist(train_dataset.pos[:, FOI], histtype='step', label='Positive train data', density=True)
plt.hist(train_dataset.neg[:, FOI], histtype='step', label='Negative train data', density=True)
plt.legend()
plt.show()

# %% [markdown]
# # Diskriminativní klasifikátory
# V následující části budete pomocí (lineární) logistické regrese přímo modelovat posteriorní pravděpodobnost třídy 1.
# Modely budou založeny čistě na NumPy, takže nemusíte instalovat nic dalšího.
# Nabitějších toolkitů se dočkáte ve třetím projektu.

# %%
def logistic_sigmoid(x):
    return np.exp(-np.logaddexp(0, -x))

def binary_cross_entropy(probs, targets):
    return np.sum(-targets * np.log(probs) - (1-targets)*np.log(1-probs)) 

class LogisticRegressionNumpy:
    def __init__(self, dim):
        self.w = np.array([0.0] * dim)
        self.b = np.array([0.0])
        
    def prob_class_1(self, x):
        return logistic_sigmoid(x @ self.w + self.b)

# %% [markdown]
# Diskriminativní klasifikátor očekává, že dostane vstup ve tvaru `[N, F]`.
# Pro práci na jediném příznaku bude tedy zapotřebí vyřezávat příslušná data v správném formátu (`[N, 1]`). 
# Doimplementujte třídu `FeatureCutter` tak, aby to zařizovalo volání její instance.
# Který příznak se použije, nechť je konfigurováno při konstrukci.
# 
# Může se Vám hodit `np.newaxis`.
# 
# **(2 řádky)**

# %%
class FeatureCutter:
    def __init__(self, fea_id):
        self.fea_id = fea_id
        
    def __call__(self, x):
      return x[:, self.fea_id][np.newaxis].transpose()

# %% [markdown]
# Dalším krokem je implementovat funkci, která model vytvoří a natrénuje.
# Jejím výstupem bude (1) natrénovaný model, (2) průběh trénovací loss a (3) průběh validační přesnosti.
# Neuvažujte žádné minibatche, aktualizujte váhy vždy na celém trénovacím datasetu.
# Po každém kroku vyhodnoťte model na validačních datech.
# Jako model vracejte ten, který dosáhne nejlepší validační přesnosti.
# Jako loss použijte binární cross-entropii  a logujte průměr na vzorek.
# Pro výpočet validační přesnosti využijte funkci `evaluate()`.
# Oba průběhy vracejte jako obyčejné seznamy.
# 
# Doporučujeme dělit efektivní learning rate počtem vzorků, na nichž je počítana loss.
# 
# **(cca 11 řádků)**

# %%
def train_logistic_regression(nb_epochs, lr, in_dim, fea_preprocessor):
    model = LogisticRegressionNumpy(in_dim)
    best_model = copy.deepcopy(model)
    losses = []
    accuracies = []
    
    train_X = fea_preprocessor(train_dataset.xs)
    train_t = train_dataset.targets

    val_X = fea_preprocessor(val_dataset.xs)

    best_acc = evaluate(best_model, val_X, val_dataset.targets)
    lr = lr/len(train_X)
    
    for _ in range(nb_epochs):
        sigmoid = model.prob_class_1(train_X)
        # logistic sigmoid weight input
        model.w = model.w - lr * (sigmoid - train_t)@(train_X)
        model.b = model.b - lr * np.sum(sigmoid - train_t)
        curr_acc = evaluate(model, val_X, val_dataset.targets)
        accuracies.append(curr_acc)
        losses.append(binary_cross_entropy(sigmoid,train_t)/(len(train_X)))
        if (curr_acc > best_acc):
            best_model = copy.deepcopy(model)
            best_acc = curr_acc

    return best_model, losses, accuracies

# %% [markdown]
# Funkci zavolejte a natrénujte model.
# Uveďte zde parametry, které vám dají slušný výsledek.
# Měli byste dostat přesnost srovnatelnou s generativním klasifikátorem s nastavenými priory.
# Neměli byste potřebovat víc, než 100 epoch.
# Vykreslete průběh trénovací loss a validační přesnosti, osu x značte v epochách.
# 
# V druhém grafu vykreslete histogramy trénovacích dat a pravděpodobnost třídy 1 pro x od -0.5 do 1.5, podobně jako výše u generativních klasifikátorů.
# 
# **(1 + 5 + 8 řádků)**

# %%
disc_fea5, losses, accuracies = train_logistic_regression(100, 3, 1, FeatureCutter(FOI))

plt.figure()
plt.xlabel('Epochs')
plt.ylabel('Value')
plt.plot( losses, label='Losses')
plt.plot(accuracies, label='Accuracies')
plt.legend()
print('w', disc_fea5.w.item(), 'b', disc_fea5.b.item())

plt.figure()
plt.xlabel('Value')
plt.ylabel('Density')
x_size = np.linspace(-0.5, 1.5, 1000)
x_size = x_size[:, np.newaxis]
plt.plot(x_size, disc_fea5.prob_class_1(x_size), label='Positive, model')
plt.hist(train_dataset.pos[:, FOI], histtype='step', label='Positive train data', density=True)
plt.hist(train_dataset.neg[:, FOI], histtype='step', label='Negative train data', density=True)
plt.legend()

print('disc_fea5:', evaluate(disc_fea5, val_dataset.xs[:, FOI][:, np.newaxis], val_dataset.targets))

# %% [markdown]
# ## Všechny vstupní příznaky
# V posledním cvičení natrénujete logistickou regresi, která využije všechn sedm vstupních příznaků.
# Zavolejte funkci z předchozího cvičení, opět vykreslete průběh trénovací loss a validační přesnosti.
# Měli byste se dostat nad 90 % přesnosti.
# 
# Může se Vám hodit `lambda` funkce.
# 
# **(1 + 5 řádků)**

# %%
disc_full_fea, losses, accuracies = train_logistic_regression(1000, 0.001, 7, lambda x:x)

plt.xlabel('Epochs')
plt.ylabel('Value')
plt.plot(losses, label='Losses')
plt.plot(accuracies, label='Accuracies')
plt.legend()
plt.show()

# %% [markdown]
# # Závěrem
# Konečně vyhodnoťte všech pět vytvořených klasifikátorů na testovacích datech.
# Stačí doplnit jejich názvy a předat jim odpovídající příznaky.
# Nezapomeňte, že u logistické regrese musíte zopakovat formátovací krok z `FeatureCutter`u.

# %%
xs_full = test_dataset.xs
xs_foi = test_dataset.xs[:, FOI]
targets = test_dataset.targets

print('Baseline:', evaluate(baseline, xs_full, targets))
print('Generative classifier (w/o prior):', evaluate(classifier_flat_prior, xs_foi, targets))
print('Generative classifier (correct):', evaluate(classifier_full_prior, xs_foi, targets))
print('Logistic regression:', evaluate(disc_fea5, xs_foi[:, np.newaxis], targets))
print('logistic regression all features:', evaluate(disc_full_fea, xs_full, targets))

# %% [markdown]
# Blahopřejeme ke zvládnutí projektu! Nezapomeňte spustit celý notebook načisto (Kernel -> Restart & Run all) a zkontrolovat, že všechny výpočty prošly podle očekávání.
# 
# Mimochodem, vstupní data nejsou synteticky generovaná.
# Nasbírali jsme je z baseline řešení předloňského projektu; vaše klasifikátory v tomto projektu predikují, že daný hráč vyhraje dicewars, takže by se daly použít jako heuristika pro ohodnocování listových uzlů ve stavovém prostoru hry.
# Pro představu, data jsou z pozic pět kol před koncem partie pro daného hráče.
# Poskytnuté příznaky popisují globální charakteristiky stavu hry jako je například poměr délky hranic předmětného hráče k ostatním hranicím.
# Nejeden projekt v ročníku 2020 realizoval požadované "strojové učení" kopií domácí úlohy.

# %%



