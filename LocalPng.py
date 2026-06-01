import os
import gc
import time
import argparse
from datetime import datetime
import torch
from min_dalle import MinDalle
import multiprocessing  # <--- Adicionado para isolamento de memória

# ==========================================
# CONFIGURAÇÕES DE CORES PARA O TERMINAL
# ==========================================
class Colors:
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    RESET = '\033[0m'

# Limita as threads para evitar sobrecarga e superaquecimento da CPU
torch.set_num_threads(4)

def clear_screen():
    os.system('clear' if os.name == 'posix' else 'cls')

def print_header():
    clear_screen()
    print(f"{Colors.CYAN}{Colors.BOLD}")
    print(r"""
 _                    _ ____              
| |    ___   ___ __ _| |  _ \ _ __   __ _ 
| |   / _ \ / __/ _` | | |_) | '_ \ / _` |
| |__| (_) | (_| (_| | |  __/| | | | (_| |
|_____\___/ \___\__,_|_|_|   |_| |_|\__, |
                                    |___/ 
    """)
    print(f"       [ AI Image Generator - Termux ]{Colors.RESET}")
    print("=" * 50)

def setup_directories():
    """Garante que o diretório de saída exista."""
    out_dir = "outputs"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    return out_dir

def generate(prompt, seed, out_dir):
    print(f"\n{Colors.YELLOW}[INFO] Inicializando processo...{Colors.RESET}")
    
    start_time = time.time()
    
    print(f"{Colors.YELLOW}[1/2] Carregando modelo MinDalle na memória...{Colors.RESET}")
    print(f"{Colors.CYAN}DICA: Feche apps em segundo plano para não faltar RAM.{Colors.RESET}")
    
    model = None
    try:
        model = MinDalle(
            is_mega=False,
            models_root='./models',
            dtype=torch.float32, 
            device='cpu',
            is_reusable=False 
        )

        print(f"\n{Colors.YELLOW}[2/2] Processando a imagem... (Isso pode demorar bastante){Colors.RESET}")
        
        image = model.generate_image(
            text=prompt,
            seed=seed,
            grid_size=1,
        )

        # Formatação do nome do arquivo
        nome_limpo = "".join(c for c in prompt.replace(" ", "_") if c.isalnum() or c == '_').lower()
        nome_limpo = nome_limpo[:40] if nome_limpo else "resultado"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        nome_arquivo = f"{nome_limpo}_{timestamp}.png"
        caminho_completo = os.path.join(out_dir, nome_arquivo)
        
        image.save(caminho_completo)
        
        elapsed_time = time.time() - start_time
        mins, secs = divmod(elapsed_time, 60)

        print(f"\n{Colors.GREEN}{Colors.BOLD}🎉 Sucesso! Imagem gerada em {int(mins)}m {int(secs)}s{Colors.RESET}")
        print(f"{Colors.GREEN}📁 Salvo em: {os.path.abspath(caminho_completo)}{Colors.RESET}")
        
    except MemoryError:
        print(f"\n{Colors.RED}{Colors.BOLD}❌ ERRO FATAL: Falta de Memória RAM!{Colors.RESET}")
        print(f"{Colors.RED}O Termux matou o processo. Reinicie o celular ou feche mais aplicativos.{Colors.RESET}")
    except Exception as e:
        print(f"\n{Colors.RED}{Colors.BOLD}❌ ERRO INESPERADO:{Colors.RESET} {e}")
    finally:
        print(f"\n{Colors.CYAN}[INFO] Finalizando subset...{Colors.RESET}")
        if model is not None:
            del model
        gc.collect()

def main():
    # Configura o método 'spawn' para o PyTorch não travar no Termux ao criar subprocessos
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser(description="Gere imagens com MinDalle pelo Termux.")
    parser.add_argument("-p", "--prompt", type=str, help="O prompt da imagem (em inglês).")
    parser.add_argument("-s", "--seed", type=int, default=-1, help="Seed para geração (padrão: -1 para aleatório).")
    args = parser.parse_args()

    out_dir = setup_directories()

    # Modo Direto (via argumentos)
    if args.prompt:
        print_header()
        print(f"{Colors.CYAN}Prompt:{Colors.RESET} {args.prompt}")
        print(f"{Colors.CYAN}Seed:{Colors.RESET} {args.seed}")
        
        # Executa em processo isolado
        p = multiprocessing.Process(target=generate, args=(args.prompt, args.seed, out_dir))
        p.start()
        p.join()
        
        print("=" * 50)
        return

    # Modo Interativo (Loop)
    while True:
        print_header()
        prompt = input(f"\n{Colors.BOLD}Digite o prompt{Colors.RESET} (Inglês) ou 'sair': ").strip()
        
        if prompt.lower() in ['sair', 'exit', 'quit', 'q']:
            print(f"\n{Colors.GREEN}Encerrando o programa. Até a próxima!{Colors.RESET}")
            break
            
        if not prompt:
            print(f"{Colors.RED}Prompt vazio! Tente novamente.{Colors.RESET}")
            time.sleep(1)
            continue

        seed_input = input(f"Digite a {Colors.BOLD}seed{Colors.RESET} (Deixe em branco para aleatório): ").strip()
        seed = int(seed_input) if seed_input.isdigit() else -1

        # Cria, roda e fecha o processo de geração para limpar o cache de RAM completamente
        processo_geracao = multiprocessing.Process(target=generate, args=(prompt, seed, out_dir))
        processo_geracao.start()
        processo_geracao.join() # Aguarda a geração terminar antes de liberar o menu de novo
        
        input(f"\n{Colors.BOLD}Pressione [ENTER] para gerar outra imagem...{Colors.RESET}")

if __name__ == "__main__":
    main()
