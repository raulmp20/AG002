from Database.banco import Banco
from IA.Artificial_Intelligence import Inteligencia

host = '127.0.0.1'
user = 'root'
password = 'root'

db = Banco(host, user, password)

db.criarBanco()
print(db.procurarDados())

ia = Inteligencia(db.procurarDados())
ia.resultados_treinos()

dados = db.inserirDados()

ia.evaluate_client(dados)
