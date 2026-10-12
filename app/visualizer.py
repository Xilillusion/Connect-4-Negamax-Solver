import time
import threading
from collections import deque
from typing import Optional
import pygame
from OpenGL.GL import *
from pygame.locals import *
from solver.negamax_zobrist import StatNegamaxSolver
from solver.connect4_array import Connect4


class SearchStats:
    """搜索统计信息类，独立管理所有统计数据"""
    def __init__(self, num_cols: int = 7):
        self.num_cols = num_cols
        self.reset()
    
    def reset(self):
        """重置所有统计"""
        self.total_nodes = 0
        self.hash_hits = 0
        self.hash_lookups = 0
        self.current_depth = 0
        self.max_depth = 0
        self.branch_depths = {i: [] for i in range(self.num_cols)}
        self.start_time = time.time()
        self.lock = threading.Lock()
    
    def update(self, depth: int, move: Optional[int], is_hash_hit: bool):
        """更新统计数据"""
        with self.lock:
            self.total_nodes += 1
            self.current_depth = depth
            self.max_depth = max(self.max_depth, depth)
            self.hash_lookups += 1
            
            if is_hash_hit:
                self.hash_hits += 1
            
            if move is not None and move < self.num_cols:
                self.branch_depths[move].append(depth)
    
    def get_hit_rate(self) -> float:
        """获取命中率百分比"""
        if self.hash_lookups == 0:
            return 0
        return self.hash_hits / self.hash_lookups * 100
    
    def get_nps(self) -> float:
        """获取每秒节点数"""
        elapsed = time.time() - self.start_time
        if elapsed == 0:
            return 0
        return self.total_nodes / elapsed


class VisualizerUI:
    """使用pygame和OpenGL显示实时搜索统计"""
    WIDTH = 1400
    HEIGHT = 800

    def __init__(self, solver: StatNegamaxSolver, stats: SearchStats):
        self.solver = solver
        self.stats = stats
        pygame.init()
        pygame.font.init()
        self.screen = pygame.display.set_mode(
            (self.WIDTH, self.HEIGHT), DOUBLEBUF | OPENGL
        )
        pygame.display.set_caption("Negamax Search Visualizer")
        self.font_large = pygame.font.Font(None, 28)
        self.font_small = pygame.font.Font(None, 20)
        self.init_gl()
        
        # 历史数据（最多保留300个采样点，即30秒@100ms）
        self.depth_history = deque(maxlen=300)
        self.nps_history = deque(maxlen=300)
        self.hit_rate_history = deque(maxlen=300)
        
        self.last_update = time.time()
        self.running = True

    def init_gl(self):
        glViewport(0, 0, self.WIDTH, self.HEIGHT)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0, self.WIDTH, self.HEIGHT, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)

    def draw_rect(self, x, y, w, h, color):
        glColor4f(*color)
        glBegin(GL_QUADS)
        glVertex2f(x, y)
        glVertex2f(x + w, y)
        glVertex2f(x + w, y + h)
        glVertex2f(x, y + h)
        glEnd()

    def draw_line(self, x1, y1, x2, y2, color, width=2):
        glColor4f(*color)
        glLineWidth(width)
        glBegin(GL_LINES)
        glVertex2f(x1, y1)
        glVertex2f(x2, y2)
        glEnd()

    def draw_text(self, text, x, y, color=(1.0, 1.0, 1.0), font=None):
        if font is None:
            font = self.font_small
        surface = font.render(text, True, (int(color[0]*255), int(color[1]*255), int(color[2]*255)))
        data = pygame.image.tostring(surface, "RGBA", True)
        glRasterPos2i(int(x), int(y))
        glDrawPixels(surface.get_width(), surface.get_height(), GL_RGBA, GL_UNSIGNED_BYTE, data)

    def draw_chart(self, x, y, width, height, data, max_val, title, color, label_func=None):
        """绘制折线图"""
        # 背景
        self.draw_rect(x, y, width, height, (0.05, 0.05, 0.1, 0.6))
        
        # 边框
        self.draw_line(x, y, x + width, y, (0.5, 0.5, 0.5, 0.8), 1)
        self.draw_line(x + width, y, x + width, y + height, (0.5, 0.5, 0.5, 0.8), 1)
        self.draw_line(x + width, y + height, x, y + height, (0.5, 0.5, 0.5, 0.8), 1)
        self.draw_line(x, y + height, x, y, (0.5, 0.5, 0.5, 0.8), 1)
        
        # 标题
        self.draw_text(title, x + 10, y + 5, color=(1.0, 1.0, 1.0), font=self.font_small)
        
        # 数据点
        if len(data) > 1:
            for i in range(len(data) - 1):
                x1 = x + 10 + (i / max(1, len(data) - 1)) * (width - 20)
                y1 = y + height - 30 - (data[i] / max_val * (height - 40)) if max_val > 0 else y + height - 30
                
                x2 = x + 10 + ((i + 1) / max(1, len(data) - 1)) * (width - 20)
                y2 = y + height - 30 - (data[i + 1] / max_val * (height - 40)) if max_val > 0 else y + height - 30
                
                self.draw_line(x1, y1, x2, y2, color, 2)
        
        # 显示最后一个值
        if len(data) > 0:
            last_val = data[-1]
            if label_func:
                label = label_func(last_val)
            else:
                label = f"{last_val:.1f}"
            self.draw_text(label, x + width - 80, y + height - 20, color=(1.0, 1.0, 1.0))

    def draw_bar_chart(self, x, y, width, height, data_dict, title, color):
        """绘制柱状图（分枝分布）"""
        # 背景
        self.draw_rect(x, y, width, height, (0.05, 0.05, 0.1, 0.6))
        
        # 边框
        self.draw_line(x, y, x + width, y, (0.5, 0.5, 0.5, 0.8), 1)
        self.draw_line(x + width, y, x + width, y + height, (0.5, 0.5, 0.5, 0.8), 1)
        self.draw_line(x + width, y + height, x, y + height, (0.5, 0.5, 0.5, 0.8), 1)
        self.draw_line(x, y + height, x, y, (0.5, 0.5, 0.5, 0.8), 1)
        
        # 标题
        self.draw_text(title, x + 10, y + 5, color=(1.0, 1.0, 1.0), font=self.font_small)
        
        # 找到最大值
        max_val = max([len(v) for v in data_dict.values()]) if data_dict else 1
        if max_val == 0:
            max_val = 1
        
        # 绘制柱子
        col_width = (width - 20) / 7
        for col in range(7):
            col_height = len(data_dict.get(col, [])) / max_val * (height - 40)
            col_x = x + 10 + col * col_width
            col_y = y + height - 30 - col_height
            
            # 选择颜色
            col_color = color if col % 2 == 0 else (color[0] * 0.7, color[1] * 0.7, color[2] * 0.7, color[3])
            self.draw_rect(col_x, col_y, col_width - 2, col_height, col_color)
            
            # 列号标签
            self.draw_text(str(col + 1), col_x + col_width / 2 - 5, y + height - 20, color=(0.8, 0.8, 0.8))

    def calculate_stats(self):
        """计算统计数据"""
        stats = self.stats
        
        # 命中率
        hit_rate = 0
        if stats.hash_lookups > 0:
            hit_rate = stats.get_hit_rate()
        
        # Nodes Per Second
        nps = stats.get_nps()
        
        return {
            'hit_rate': hit_rate,
            'nps': nps,
        }

    def update(self):
        """采样并更新历史数据（100ms采样间隔）"""
        current_time = time.time()
        if current_time - self.last_update < 0.1:
            return
        
        self.last_update = current_time
        stats_data = self.calculate_stats()
        
        with self.stats.lock:
            self.depth_history.append(self.stats.current_depth)
            self.nps_history.append(stats_data['nps'])
            self.hit_rate_history.append(stats_data['hit_rate'])

    def draw(self):
        """绘制所有图表"""
        glClearColor(0.02, 0.05, 0.14, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)
        
        self.update()
        
        with self.stats.lock:
            # 标题
            glColor4f(1.0, 1.0, 1.0, 1.0)
            self.draw_text("Negamax Search Visualizer", 20, 20, font=self.font_large)
            
            # 深度折线图
            max_depth = max(self.depth_history) if self.depth_history else 30
            self.draw_chart(20, 80, 330, 220, list(self.depth_history), max(max_depth, 1), 
                           "Search Depth Over Time", (0.3, 0.8, 1.0, 0.8))
            
            # 节点速率 (Nodes Per Second)
            max_nps = max(self.nps_history) if self.nps_history else 1000
            self.draw_chart(370, 80, 330, 220, list(self.nps_history), max(max_nps, 1),
                           "Nodes Per Second", (0.8, 0.3, 1.0, 0.8),
                           label_func=lambda x: f"{x:.0f}")
            
            # 命中率
            self.draw_chart(720, 80, 330, 220, list(self.hit_rate_history), 100,
                           "Hash Hit Rate (%)", (0.3, 1.0, 0.3, 0.8),
                           label_func=lambda x: f"{x:.1f}%")
            
            # 分枝分布图
            self.draw_bar_chart(20, 320, 330, 220, self.stats.branch_depths,
                              "Branch Exploration Depth", (1.0, 0.7, 0.3, 0.8))
            
            # 哈希表信息
            info_y = 320
            self.draw_text(f"Hash Lookups: {self.stats.hash_lookups:,}", 720, info_y, color=(0.9, 0.9, 0.9))
            self.draw_text(f"Hash Hits: {self.stats.hash_hits:,}", 720, info_y + 30, color=(0.3, 1.0, 0.3))
            self.draw_text(f"Trans Table Size: {len(self.solver.trans_table):,}", 720, info_y + 60, color=(1.0, 1.0, 0.5))

    def run(self):
        """主循环"""
        clock = pygame.time.Clock()
        while self.running:
            for event in pygame.event.get():
                if event.type == QUIT:
                    self.running = False
                elif event.type == KEYDOWN:
                    if event.key == K_ESCAPE:
                        self.running = False
            
            self.draw()
            pygame.display.flip()
            clock.tick(30)

        pygame.quit()


def visualize_search(game: Connect4):
    """便利函数：可视化一次搜索过程"""
    solver = StatNegamaxSolver(game)
    stats = solver.stats
    visualizer_ui = VisualizerUI(solver, stats)
    
    # 在后台线程执行搜索
    def search_thread():
        try:
            stats.reset()
            result = solver.negamax(alpha=-1, beta=1)
            print(f"Search completed. Best score: {result[0]}, Best move: {result[1]}")
            time.sleep(0.5)
        except Exception as e:
            print(f"Search error: {e}")
            import traceback
            traceback.print_exc()
    
    search_thread = threading.Thread(target=search_thread, daemon=True)
    search_thread.start()
    
    # 运行UI
    visualizer_ui.run()


if __name__ == "__main__":
    # 测试：可视化一个搜索过程
    board = [
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]
    game = Connect4(board)
    visualize_search(game)
