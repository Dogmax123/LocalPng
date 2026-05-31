import os
import gc
import torch
from min_dalle import MinDalle

# OTIMIZAÇÃO 1: Limita os núcleos da CPU para evitar que o celular congele ou superaqueça.
torch.set_num_threads(4)

def main():
    print("=" * 50)
    # Usando aspas triplas para imprimir o bloco ASCII inteiro perfeitamente
    print(""" _                    _ ____              
| |    ___   ___ __ _| |  _ \\ _ __   __ _ 
| |   / _ \\ / __/ _` | | |_) | '_ \\ / _` |
| |__| (_) | (_| (_| | |  __/| | | | (_| |
|_____\\___/ \\___\\__,_|_|_|   |_| |_|\\__, |
                                    |___/ """)
    print("=" * 50)

    prompt = input("\nDigite o prompt (Em inglês funciona melhor. Ex: 'cursed shrek'): ").strip()
    
    if not prompt:
        print("Nenhum prompt digitado. Encerrando.")
        return

    print("\n[1/2] Carregando o modelo na memória...")
    print("DICA: Feche todos os outros apps em segundo plano para liberar RAM!")
    
    try:
        # OTIMIZAÇÃO 2: is_reusable=False ajuda a economizar memória
        model = MinDalle(
            is_mega=False,
            models_root='./models',
            dtype=torch.float32, 
            device='cpu',
            is_reusable=False 
        )

        print("\n[2/2] Processando os pixels distorcidos...")
        print("Nota: O Termux vai congelar por alguns minutos. Deixe o celular quietinho...")

        # Gerando a imagem
        image = model.generate_image(
            text=prompt,
            seed=-1,
            grid_size=1,
        )

        # OTIMIZAÇÃO DE NOME: Transforma o prompt em um nome de arquivo válido
        nome_limpo = "".join(c for c in prompt.replace(" ", "_") if c.isalnum() or c == '_').lower()
        
        # Garante que o nome não fique vazio e limita o tamanho
        nome_limpo = nome_limpo[:50] if nome_limpo else "resultado"
        nome_arquivo = f"{nome_limpo}.png"
        
        image.save(nome_arquivo)

        print(f"\n🎉 Sucesso! Imagem salva como: {os.path.abspath(nome_arquivo)}")
        
    except MemoryError:
        print("\n❌ ERRO: Faltou memória RAM! Tente fechar mais apps ou reiniciar o celular antes de rodar.")
    except Exception as e:
        print(f"\n❌ ERRO INESPERADO: {e}")
    finally:
        # OTIMIZAÇÃO 3: Força a limpeza da RAM antes do script fechar
        if 'model' in locals():
            del model
        gc.collect()

    print("=" * 50)

if __name__ == "__main__":
    main()
        
