import os
import gc
import torch
from min_dalle import MinDalle

# OTIMIZAÇÃO 1: Limita os núcleos da CPU para evitar que o celular congele ou superaqueça.
# Se o Termux ainda fechar sozinho, tente diminuir esse número para 2.
torch.set_num_threads(4)

def main():
    print("=" * 50)
print(" _                    _ ____              "
print("| |    ___   ___ __ _| |  _ \ _ __   __ _ "
print("| |   / _ \ / __/ _` | | |_) | '_ \ / _` |"
print("| |__| (_) | (_| (_| | |  __/| | | | (_| |"
print("|_____\___/ \___\__,_|_|_|   |_| |_|\__, |"
print("                                    |___/ "
                                           
    print("=" * 50)

    prompt = input("\nDigite o prompt (Em inglês funciona melhor. Ex: 'cursed shrek'): ").strip()
    
    if not prompt:
        print("Nenhum prompt digitado. Encerrando.")
        return

    print("\n[1/2] Carregando o modelo na memória...")
    print("DICA: Feche todos os outros apps em segundo plano para liberar RAM!")
    
    try:
        # OTIMIZAÇÃO 2: is_reusable=False ajuda a economizar memória se você só vai gerar 1 imagem por vez
        model = MinDalle(
            is_mega=False,
            models_root='./models',
            dtype=torch.float32, 
            device='cpu',
            is_reusable=False 
        )

        print("\n[2/2] Processando os pixels distorcidos...")
        print("Nota: O Termux vai congelar por alguns minutos. Deixe o celular quietinho processando...")

        # Gerando a imagem
        image = model.generate_image(
            text=prompt,
            seed=-1,
            grid_size=1,
        )

        # Salvando o resultado
        nome_arquivo = "resultado_distorcido.png"
        image.save(nome_arquivo)

        print(f"\n🎉 Sucesso! Imagem salva como: {os.path.abspath(nome_arquivo)}")
        
    except MemoryError:
        print("\n❌ ERRO: Faltou memória RAM! Tente fechar mais apps ou reiniciar o celular antes de rodar.")
    except Exception as e:
        print(f"\n❌ ERRO INESPERADO: {e}")
    finally:
        # OTIMIZAÇÃO 3: Força a limpeza da RAM antes do script fechar completamente
        if 'model' in locals():
            del model
        gc.collect()

    print("=" * 50)

if __name__ == "__main__":
    main()
  
