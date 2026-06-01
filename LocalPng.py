import os
import gc
import sys
import time
import random
import argparse
from datetime import datetime
import torch
from min_dalle import MinDalle

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

# ==========================================
# CLASSE PRINCIPAL DO GERADOR
# ==========================================
class LocalPNG:
    def __init__(self, is_reusable: bool = False):
        # Limita as threads para evitar sobrecarga e superaquecimento da CPU
        torch.set_num_threads(4)
        self.out_dir = "outputs"
        self.is_reusable = is_reusable
        self.model = None
        self._setup_directories()

    def _setup_directories(self) -> None:
        """Garante que o diretório de saída exista."""
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)

    def _clear_screen(self) -> None:
        """Limpa o terminal de forma multiplataforma."""
        os.system('clear' if os.name == 'posix' else 'cls')

    def print_header(self) -> None:
        self._clear_screen()
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
        if self.is_reusable:
            print(f"{Colors.YELLOW}⚡ Modo Rápido (Reutilizável) ATIVADO - Consome mais RAM{Colors.RESET}")
        print("=" * 50)

    def _load_model(self) -> None:
        """Carrega o modelo apenas se necessário."""
        if self.model is None:
            print(f"{Colors.YELLOW}[*] Carregando modelo MinDalle na memória...{Colors.RESET}")
            print(f"{Colors.CYAN}DICA: Feche apps em segundo plano para liberar RAM.{Colors.RESET}")
            self.model = MinDalle(
                is_mega=False,
                models_root='./models',
                dtype=torch.float32, 
                device='cpu',
                is_reusable=self.is_reusable 
            )

    def generate(self, prompt: str, seed: int) -> None:
        print(f"\n{Colors.YELLOW}[INFO] Inicializando processo para: '{prompt}'...{Colors.RESET}")
        start_time = time.time()
        
        try:
            self._load_model()

            print(f"\n{Colors.YELLOW}[*] Renderizando a imagem... (Pode demorar bastante na CPU){Colors.RESET}")
            
            image = self.model.generate_image(
                text=prompt,
                seed=seed,
                grid_size=1,
            )

            # Formatação segura do nome do arquivo
            nome_limpo = "".join(c for c in prompt.replace(" ", "_") if c.isalnum() or c == '_').lower()
            nome_limpo = nome_limpo[:30] if nome_limpo else "resultado"
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            nome_arquivo = f"{nome_limpo}_{timestamp}_s{seed}.png"
            caminho_completo = os.path.join(self.out_dir, nome_arquivo)
            
            image.save(caminho_completo)
            
            elapsed_time = time.time() - start_time
            mins, secs = divmod(elapsed_time, 60)

            print(f"\n{Colors.GREEN}{Colors.BOLD}🎉 Sucesso! Imagem gerada em {int(mins)}m {int(secs)}s{Colors.RESET}")
            print(f"{Colors.GREEN}📁 Salvo em: {os.path.abspath(caminho_completo)}{Colors.RESET}")
            
        except MemoryError:
            print(f"\n{Colors.RED}{Colors.BOLD}❌ ERRO FATAL: Out Of Memory (OOM)!{Colors.RESET}")
            print(f"{Colors.RED}O Termux esgotou a RAM. Reinicie o script ou feche outros apps.{Colors.RESET}")
        except Exception as e:
            print(f"\n{Colors.RED}{Colors.BOLD}❌ ERRO INESPERADO:{Colors.RESET} {e}")
        finally:
            # Se não for reutilizável, limpa a memória imediatamente
            if not self.is_reusable:
                print(f"\n{Colors.CYAN}[INFO] Liberando memória RAM...{Colors.RESET}")
                if self.model is not None:
                    del self.model
                    self.model = None
                gc.collect()

# ==========================================
# FLUXO DE EXECUÇÃO (CLI / INTERATIVO)
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Local PNG - Gerador de Imagens AI no Termux.")
    parser.add_argument("-p", "--prompt", type=str, help="O prompt da imagem (em inglês).")
    parser.add_argument("-s", "--seed", type=int, help="Seed fixa para geração (opcional).")
    parser.add_argument("-r", "--reusable", action="store_true", help="Mantém o modelo na RAM para gerar várias imagens mais rápido.")
    args = parser.parse_args()

    app = LocalPNG(is_reusable=args.reusable)

    try:
        # Modo Direto (via argumentos do terminal)
        if args.prompt:
            app.print_header()
            seed = args.seed if args.seed is not None else random.randint(0, 2**31 - 1)
            print(f"{Colors.CYAN}Prompt:{Colors.RESET} {args.prompt}")
            print(f"{Colors.CYAN}Seed:{Colors.RESET} {seed}")
            app.generate(args.prompt, seed)
            print("=" * 50)
            return

        # Modo Interativo (Loop)
        while True:
            app.print_header()
            prompt = input(f"\n{Colors.BOLD}Digite o prompt{Colors.RESET} (Inglês) ou 'sair': ").strip()
            
            if prompt.lower() in ['sair', 'exit', 'quit', 'q']:
                print(f"\n{Colors.GREEN}Encerrando o Local PNG. Até mais!{Colors.RESET}")
                break
                
            if not prompt:
                print(f"{Colors.RED}Prompt vazio! Tente novamente.{Colors.RESET}")
                time.sleep(1.5)
                continue

            seed_input = input(f"Digite a {Colors.BOLD}seed{Colors.RESET} (Pressione ENTER para aleatório): ").strip()
            
            # Correção do bug do -1: Geração de seed aleatória real
            if seed_input.isdigit():
                seed = int(seed_input)
            else:
                seed = random.randint(0, 2**31 - 1)

            app.generate(prompt, seed)
            
            input(f"\n{Colors.BOLD}Pressione [ENTER] para continuar...{Colors.RESET}")

    except KeyboardInterrupt:
        print(f"\n\n{Colors.RED}Operação cancelada pelo usuário (Ctrl+C). Saindo de forma segura...{Colors.RESET}")
        sys.exit(0)

if __name__ == "__main__":
    main()
