import { create } from 'zustand'
import { api } from '../services/api'
import toast from 'react-hot-toast'

export interface Session {
  id: string
  title: string
  created_at: string
  updated_at: string
}

const SESSIONS_PER_PAGE = 15;

interface ChatState {
  sessions: Session[];
  sessionPage: number;
  sessionTotal: number;
  fetchSessions: () => Promise<void>;
  setSessionPage: (page: number) => void;
  deleteSession: (sessionId: string) => Promise<void>;
}

export const useChatStore = create<ChatState>((set, get) => ({
  sessions: [],
  sessionPage: 1,
  sessionTotal: 0,

  setSessionPage: (page: number) => {
    set({ sessionPage: page });
  },

  fetchSessions: async () => {
    const { sessionPage } = get();
    try {
      const res = await api.get(`/chat/sessions?page=${sessionPage}&limit=${SESSIONS_PER_PAGE}`);
      set({ sessions: res.data.sessions || [], sessionTotal: res.data.total || 0 });
    } catch (e: any) {
      toast.error(e?.response?.data?.detail || 'Không thể tải lịch sử chat');
      set({ sessions: [], sessionTotal: 0 });
    }
  },

  deleteSession: async (sessionId: string) => {
    try {
      await api.delete(`/chat/sessions/${sessionId}`);
      toast.success('Đã xóa phiên chat');

      const { sessions, sessionPage } = get();
      // If the deleted session was the last one on the current page, go back one page.
      if (sessions.length === 1 && sessionPage > 1) {
        set({ sessionPage: sessionPage - 1 });
      }
      // Refetch the (potentially new) current page.
      get().fetchSessions();

    } catch (e: any) {
      toast.error(e?.response?.data?.detail || 'Không thể xóa phiên chat');
    }
  },
}));

