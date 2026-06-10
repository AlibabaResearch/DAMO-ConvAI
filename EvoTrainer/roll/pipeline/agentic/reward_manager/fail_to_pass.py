import re


class FailToPassJudge:
    """
    和query相关的单测要执行正确
    """

    def parse_pytest_summary(self, output):
        """
        解析 pytest 输出摘要
        
        支持的状态: passed, failed, skipped, error, xfailed, xpassed, subtests passed, warning(s) 等
        """
        # 从后往前找包含 "in" 和时间的摘要行
        lines = output.strip().split('\n')
        summary_line = None
        
        # 查找包含 "in X.XXs" 或 "in X.XX seconds" 的行
        for line in reversed(lines):
            # 必须包含 passed/failed/skipped 等关键词之一，避免匹配到其他行
            if ' in ' in line and re.search(r'\d+\.?\d*s', line):
                # 检查是否包含测试状态关键词
                status_keywords = ['passed', 'failed', 'skipped', 'error', 'warning', 'xfailed', 'xpassed']
                if any(keyword in line.lower() for keyword in status_keywords):
                    summary_line = line
                    break
        
        if not summary_line:
            return None
        
        # 移除等号装饰符
        summary_line = re.sub(r'=+', '', summary_line).strip()
        
        # 匹配时间：支持 "in X.XXs" 或 "in X.XX seconds"
        time_pattern = r'in\s+([\d.]+)\s*s(?:econds)?'
        time_match = re.search(time_pattern, summary_line)
        
        if not time_match:
            return None
        
        duration = float(time_match.group(1))
        
        # 提取时间之前的部分作为摘要文本
        summary_text = summary_line[:time_match.start()].strip()
        
        # 匹配状态：数字 + 单词（支持单复数）
        # 改进：支持 "1 warning" 和 "2 warnings" 等单复数形式
        status_pattern = r'(\d+)\s+([\w\s]+?)(?=,|\s*$)'
        matches = re.findall(status_pattern, summary_text)
        
        result = {
            'duration': duration,
            'summary': summary_text.strip()
        }
        
        # 添加各状态计数
        for count, status in matches:
            # 规范化状态名
            status = status.strip()
            # 统一单复数：去掉末尾的 's'（如果存在）
            normalized_status = status.lower()
            
            # 将 "warning" 和 "warnings" 都统一为 "warnings"
            # 将 "error" 和 "errors" 都统一为 "error"
            if normalized_status == 'warning':
                normalized_status = 'warnings'
            elif normalized_status == 'errors':
                normalized_status = 'error'
            
            # 替换空格为下划线（处理 "subtests passed" 等）
            normalized_status = normalized_status.replace(' ', '_')
            
            result[normalized_status] = int(count)
        
        # 计算总测试数（不包括 warnings）
        test_statuses = ['passed', 'failed', 'skipped', 'error', 'xfailed', 'xpassed', 'subtests_passed']
        result['total'] = sum(result.get(status, 0) for status in test_statuses)
        
        return result