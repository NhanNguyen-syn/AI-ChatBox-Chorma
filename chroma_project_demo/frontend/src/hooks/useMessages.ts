import { useState, useEffect } from 'react';
import { api } from '../services/api';
import toast from 'react-hot-toast';

interface ChatMessageItem {
  id: string;
  message: string;
  response: string;
  timestamp: string;
  session_id: string;
}

export const useMessages = (sessionId: string | null) => {
  const [messages, setMessages] = useState<ChatMessageItem[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!sessionId) {
      setMessages([]);
      return;
    }

    const fetchMessages = async () => {
      setLoading(true);
      try {
        const res = await api.get(`/chat/sessions/${sessionId}/messages`);
        setMessages(res.data || []);
      } catch (e: any) {
        toast.error(e?.response?.data?.detail || 'Không thể tải tin nhắn');
        setMessages([]);
      } finally {
        setLoading(false);
      }
    };

    fetchMessages();
  }, [sessionId]);

  return { messages, loading };
};
