package mock_framework.db;

import java.util.List;
import java.util.Map;

/**
 * 사내 표준 DB 템플릿 클래스 (In-house DB Template)
 * MyBatis 기반의 사내 레거시 DB 접근을 추상화한 유틸리티입니다.
 * 모든 비즈니스 로직은 직접 Connection을 맺지 않고 이 클래스를 사용해야 합니다.
 */
public class CompanyDbTemplate {

    /**
     * 페이징 처리가 포함된 리스트 조회 메서드입니다.
     * 사내 표준 페이징 객체인 PageRequest를 반드시 파라미터로 넘겨야 합니다.
     *
     * @param queryId 실행할 MyBatis 쿼리 ID (예: "User.selectUserList")
     * @param params 쿼리에 바인딩할 파라미터 맵
     * @param page 현재 페이지 번호 (1부터 시작)
     * @param size 페이지당 노출할 데이터 개수
     * @return 페이징 처리된 결과 데이터 리스트
     */
    public List<Map<String, Object>> selectListWithPaging(String queryId, Map<String, Object> params, int page, int size) {
        int offset = (page - 1) * size;
        params.put("company_offset", offset);
        params.put("company_limit", size);
        
        // 가상의 사내 DB 실행 로직
        // return sqlSession.selectList(queryId, params);
        return null; 
    }

    /**
     * 단건 데이터를 삽입(Insert)할 때 사용하는 사내 표준 메서드입니다.
     * 등록자(reg_id)와 등록일시(reg_dt)가 자동으로 맵핑됩니다.
     */
    public int insert(String queryId, Map<String, Object> params) {
        // 내부 공통 로직 실행 (reg_id 자동 주입 등)
        return 1;
    }
}