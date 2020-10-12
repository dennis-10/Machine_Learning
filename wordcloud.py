"""
    Natural Language Toolkit, ou mais comumente o NLTK, é um conjunto de bibliotecas e programas
 para processamento simbólico e estatístico da linguagem natural para inglês, escrito na linguagem
de programação Python.

"""
# Importação das bibliotecas
import matplotlib.pyplot as plt
import nltk # Para mineração de texto
nltk.download('stopwords') # Atualiza as stopwords(palavras removidas antes ou após o processamento de texto
from nltk.corpus import PlaintextCorpusReader
from nltk.corpus import stopwords
from matplotlib.colors import ListedColormap
from wordcloud import WordCloud
import string

# Criação de um Objeto Corpus. Lendo textos do disco
corpus = PlaintextCorpusReader('Arquivos', '.*', encoding="ISO-8859-1") # Corpus contém um ou vários arquivos textuais

# Leitura dos arquivos do disco, percorre os registros e mostra o nome dos primeiros 100 arquivos
arquivos = corpus.fileids()
# primeiro arquivo
arquivos[0]

# zero a 9
arquivos[0:10]

# Imprime todos os nomes
for a in arquivos:
    print(a)

# Acesso ao texto do primeiro arquivo
texto = corpus.raw('1.txt')
texto

# Acesso a todas as palavras de todos os arquivos do corpus
todo_texto = corpus.raw()
todo_texto

# Obtenção de todas as palavras no Corpus e visualização da quantidade
palavras = corpus.words()
# Acessando pelo índice
palavras[170]
# Quantas palavras
len(palavras)


# USando NLTK, obtemos as stopwords em inglês
stops = stopwords.words('english') # Palavras sem valor semântico
# stops = stopwords.words('portuguese') # Caso textos fossem em pt
stops

# Fazendo nuvem de palavras
# Definição as cores que serão utilizadas na nuvem de palavras
mapa_cores = ListedColormap(['orange','green','red','magenta'])
# Criação de nuvem de palavras, com no máximo 100 palavras e utilizando as top words
nuvem = WordCloud(background_color='white',
                  colormap= mapa_cores, # PAssando as cores
                  stopwords=stops,
                  max_words=100)

# Criação e visualização da nuvem de palavras
nuvem.generate(todo_texto)
plt.imshow(nuvem)

# Criação de nova lista de palavras, removendo stop words
palavras_semstop = [p for p in palavras if p not in stops] # Cria uma lista com palavra 'p' se p nao está
# em 'stop' de stopwords
len(palavras_semstop)

# Remoção da pontuação, gerando uma lista sem stop words e sem pontuação
palavras_sem_pontuacao = [p for p in palavras_semstop if p not in string.punctuation]
len(palavras_sem_pontuacao) # Removei cerca de 500 pontuações

# Cálculo da frequência das palavras e visualização das mais comuns
frequencia = nltk.FreqDist(palavras_sem_pontuacao)
frequencia

# mais comuns
maiscomuns = frequencia.most_common(100)
maiscomuns